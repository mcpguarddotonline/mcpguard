"""A dev set of agent tool outputs, built from AgentDojo (MIT).

    python evals/ml/build_dev_agent.py AGENTDOJO_PKG_DIR OUT_DIR

AGENTDOJO_PKG_DIR is the unpacked `agentdojo` package (pip download
agentdojo, unzip the wheel). Writes OUT_DIR/dev_agent.jsonl.

Round 2 showed that our other dev sets don't predict performance on agent
tool outputs, so model and threshold choices need a dev set that looks like
what a gateway sees. AgentDojo's four suites (workspace, slack, travel,
banking) hold realistic tool data (emails, files, calendar events, Slack
messages, web pages, hotel reviews, transactions) with marked slots where an
attacker could write.

* benign rows: every data item, with the slots holding AgentDojo's own
  harmless default text;
* attack rows: the same items with one slot holding an attack, built from
  AgentDojo's own injection goals and attack templates. We write no attack
  text ourselves.

AgentDojo's "injecagent" template is copied from the InjecAgent paper, which
is one of our held-out sets, so it is left out. This set is for choosing
models and thresholds only; it is never trained on.
"""

import ast
import glob
import json
import os
import random
import sys

import yaml

PKG, OUT = sys.argv[1], sys.argv[2]
random.seed(7)
SUITES = ["workspace", "slack", "travel", "banking"]
USER = {"workspace": "Emma Johnson", "slack": "Emma Johnson", "travel": "Emma Johnson", "banking": "Emma Johnson"}


class Loader(yaml.SafeLoader):
    pass


Loader.add_multi_constructor("!", lambda l, tag, node: None)


def module_strings(path):
    """Module-level string constants and every class GOAL, evaluated without
    importing the package."""
    tree = ast.parse(open(path).read())
    ns, goals = {}, []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.JoinedStr)):
            try:
                ns[node.targets[0].id] = eval(compile(ast.Expression(node.value), path, "eval"), {}, ns)
            except Exception:
                pass
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            local = dict(ns)
            for st in node.body:
                if isinstance(st, (ast.Assign, ast.AnnAssign)) and isinstance(st.value, (ast.Constant, ast.JoinedStr)):
                    tgt = st.targets[0] if isinstance(st, ast.Assign) else st.target
                    try:
                        local[tgt.id] = eval(compile(ast.Expression(st.value), path, "eval"), {}, local)
                    except Exception:
                        continue
                    if tgt.id == "GOAL":
                        goals.append(local["GOAL"])
    return goals


def templates():
    """String templates with a {goal} slot from AgentDojo's attack modules,
    skipping the class copied from InjecAgent."""
    out = []
    for f in glob.glob(f"{PKG}/attacks/*.py"):
        tree = ast.parse(open(f).read())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            names = [s.value.value for s in cls.body if isinstance(s, ast.Assign)
                     and getattr(s.targets[0], "id", "") == "name" and isinstance(s.value, ast.Constant)]
            if "injecagent" in names:
                continue
            for n in ast.walk(cls):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and "{goal}" in n.value:
                    out.append(n.value)
                elif isinstance(n, ast.JoinedStr):
                    src = ast.get_source_segment(open(f).read(), n) or ""
                    if "{goal}" in src and "{" in src:
                        pass  # f-strings are formatted at run time; plain templates cover them
    for f in glob.glob(f"{PKG}/attacks/*.py"):
        tree = ast.parse(open(f).read())
        for n in tree.body:
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) \
                    and "{goal}" in n.value.value:
                out.append(n.value.value)
    return sorted(set(out))


def fill_template(t, goal, user):
    return (t.replace("{goal}", goal).replace("{user}", user).replace("{model}", random.choice(["GPT-4", "Claude", "Gemini"]))
             .replace("{{", "{").replace("}}", "}"))


def items(obj):
    """Data items an agent would get back from a tool call: a dict with no
    nested lists of records, or one long text value."""
    if isinstance(obj, dict):
        def records(v):
            xs = v.values() if isinstance(v, dict) else v
            return any(isinstance(x, dict) or (isinstance(x, list) and any(isinstance(y, dict) for y in x)) for x in xs)

        nested = [v for v in obj.values() if isinstance(v, (dict, list)) and records(v)]
        if not nested:
            yield obj
            return
        for k, v in obj.items():
            if isinstance(v, str) and len(v) > 60:
                yield {"url" if "." in str(k) else "key": k, "content": v}
            elif isinstance(v, (dict, list)):
                yield from items(v)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, str) for x in obj) and sum(map(len, obj)) > 120:
            yield {"items": obj}
            return
        for x in obj:
            if isinstance(x, (dict, list)):
                yield from items(x)


def render(item):
    if random.random() < 0.5:
        return json.dumps(item, ensure_ascii=False, default=str)
    return yaml.safe_dump(item, allow_unicode=True, sort_keys=False, default_flow_style=False)


rows = []
for suite in SUITES:
    d = f"{PKG}/data/suites/{suite}"
    vectors = yaml.safe_load(open(f"{d}/injection_vectors.yaml"))
    goals = module_strings(f"{PKG}/default_suites/v1/{suite}/injection_tasks.py")
    tmpl = templates()
    files = [f"{d}/environment.yaml"] + sorted(glob.glob(f"{d}/include/*.yaml"))
    raw = "\n".join(open(f).read() for f in files)
    for f in files:
        text = open(f).read()
        for v in vectors:
            text = text.replace("{" + v + "}", f"QQV{v}QQ")
        try:
            data = yaml.load(text, Loader=Loader)
        except yaml.YAMLError:
            continue
        its = [json.dumps(it, ensure_ascii=False, default=str) for it in items(data)]

        def realise(group, attack_slot=None, attack=""):
            out = []
            for o in group:
                for v in vectors:
                    val = attack if v == attack_slot else (vectors[v].get("default") or "")
                    o = o.replace(f"QQV{v}QQ", json.dumps(val, ensure_ascii=False)[1:-1])
                out.append(json.loads(o))
            return render(out[0] if len(out) == 1 else out)

        # A tool call returns one item or a short list (search results).
        for it in its:
            rows.append({"text": realise([it]), "label": "benign", "source": f"agentdojo_{suite}"})
        for _ in range(3 * len(its)):
            g = random.sample(its, min(len(its), random.randint(2, 4)))
            rows.append({"text": realise(g), "label": "benign", "source": f"agentdojo_{suite}"})
        for it in its:
            for v in [v for v in vectors if f"QQV{v}QQ" in it]:
                for _ in range(10):
                    a = fill_template(random.choice(tmpl), random.choice(goals), USER[suite])
                    others = random.sample([x for x in its if x != it], min(len(its) - 1, random.randint(0, 3)))
                    g = others + [it]
                    random.shuffle(g)
                    rows.append({"text": realise(g, v, a), "label": "injection", "source": f"agentdojo_{suite}"})
    print(suite, len(vectors), "slots", len(goals), "goals", len(tmpl), "templates", file=sys.stderr)

seen, uniq = set(), []
for r in rows:
    if r["text"] not in seen:
        seen.add(r["text"])
        uniq.append(r)
random.shuffle(uniq)
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/dev_agent.jsonl", "w") as f:
    for r in uniq:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
c = {}
for r in uniq:
    c[(r["source"], r["label"])] = c.get((r["source"], r["label"]), 0) + 1
print(c)
