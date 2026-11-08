#!/usr/bin/env python3
"""Downloads public third-party injection datasets and writes them as JSONL
into evals/public/ (git-ignored). Standard library only.

    python3 evals/fetch_public.py
    cargo run --release -p mcpguard-engine --example eval -- evals/public/heldout/*.jsonl

Sources (all permissively licensed; we don't redistribute them):
  deepset/prompt-injections          Apache-2.0  direct prompts, labeled both ways
  Lakera/gandalf_ignore_instructions MIT         real attack attempts from the Gandalf game
  uiuc-kang-lab/InjecAgent           MIT         attacks hidden inside tool responses,
                                                 plus benign simulated tool responses
"""

import json
import os
import time
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
UA = {"User-Agent": "mcpguard-evals"}


def get(url, tries=8):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:  # rate limits and transient errors
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def hf_rows(dataset, split):
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": split, "offset": offset, "length": 100})
        page = json.loads(get("https://datasets-server.huggingface.co/rows?" + q))
        rows += [r["row"] for r in page["rows"]]
        offset += 100
        if offset >= page["num_rows_total"]:
            return rows
        time.sleep(1)


def write(name, items):
    seen, n = set(), {"injection": 0, "benign": 0}
    with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
        for text, label in items:
            text = text.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            n[label] += 1
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
    print(f"{name}: {n['injection']} injections, {n['benign']} benign")


def main():
    os.makedirs(OUT, exist_ok=True)

    # Rules may only be tuned on the train/ files; heldout/ is for reporting.
    os.makedirs(os.path.join(OUT, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "heldout"), exist_ok=True)

    lab = lambda r: (r["text"], "injection" if r["label"] == 1 else "benign")
    write("train/deepset", map(lab, hf_rows("deepset/prompt-injections", "train")))
    write("heldout/deepset", map(lab, hf_rows("deepset/prompt-injections", "test")))

    write("train/gandalf", ((r["text"], "injection") for r in hf_rows("Lakera/gandalf_ignore_instructions", "train")))
    held = hf_rows("Lakera/gandalf_ignore_instructions", "validation") + hf_rows("Lakera/gandalf_ignore_instructions", "test")
    write("heldout/gandalf", ((r["text"], "injection") for r in held))

    base = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/"
    attacks = []
    for f in ("test_cases_dh_base.json", "test_cases_ds_base.json"):
        attacks += [(c["Tool Response"], "injection") for c in json.loads(get(base + f))]
    benign = [(v, "benign") for v in json.loads(get(base + "attacker_simulated_responses.json")).values()]
    # Never used for tuning: rules shouldn't chase polite natural-language requests.
    write("heldout/injecagent", attacks + benign)


if __name__ == "__main__":
    main()
