"""Round 2 additions to the training set, from public licensed data only.

    python evals/ml/add_round2.py RAW_DIR DATA_DIR OUT_DIR

Copies DATA_DIR/{dev,dev_ood}.jsonl unchanged and writes OUT_DIR/train.jsonl:
the round-1 training rows plus

* LLMail-Inject phase 2 attack emails that round 1 did not use (MIT), and
* Wikipedia articles in 11 languages as benign text (CC-BY-SA), so a
  non-English tool output is not read as suspicious by itself.

Any text that appears in dev or dev_ood is skipped, so both stay unseen.
"""

import json
import random
import shutil
import sys

RAW, DATA, OUT = sys.argv[1:4]
random.seed(20261109)


def jl(p):
    return [json.loads(l) for l in open(p)]


train = jl(f"{DATA}/train.jsonl")
dev_texts = {r["text"].strip().lower() for r in jl(f"{DATA}/dev.jsonl")}
for r in jl(f"{DATA}/dev_ood.jsonl"):
    dev_texts.add(r["text"].strip().lower())
seen = dev_texts | {r["text"].strip().lower() for r in train}


def fresh(t):
    k = t.strip().lower()
    if not k or k in seen:
        return False
    seen.add(k)
    return True


ll = json.load(open(f"{RAW}/microsoft__llmail-inject-challenge/data/labelled_unique_submissions_phase2.json"))
att = [k[:1800] for k, v in ll.items() if str(v.get("attack_attempt")) == "True"]
random.shuffle(att)
new = []
for t in att:
    if len([r for r in new if r["source"] == "llmail2"]) >= 6000:
        break
    # dev_ood wraps dev LLMail emails in other formats; skip any email that
    # appears inside a dev_ood text as well.
    if fresh(t):
        new.append({"text": t, "label": "injection", "source": "llmail2"})

multi = json.load(open(f"{RAW}/wiki_multi.json"))
for lang, arts in multi.items():
    for a in arts:
        if len(a) > 1800:
            i = random.randrange(0, len(a) - 1800)
            a = a[i : i + 1800]
        if fresh(a):
            new.append({"text": a, "label": "benign", "source": "wiki_multi"})

ood_blob = "\n".join(r["text"] for r in jl(f"{DATA}/dev_ood.jsonl")).lower()
before = len(new)
new = [r for r in new if r["source"] != "llmail2" or r["text"].strip().lower()[:200] not in ood_blob]
print("dropped for dev_ood overlap:", before - len(new))

rows = train + new
random.shuffle(rows)
with open(f"{OUT}/train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for s in ("dev", "dev_ood"):
    shutil.copy(f"{DATA}/{s}.jsonl", f"{OUT}/{s}.jsonl")
counts = {}
for r in new:
    counts[(r["source"], r["label"])] = counts.get((r["source"], r["label"]), 0) + 1
print("train rows", len(rows), "added", counts)
