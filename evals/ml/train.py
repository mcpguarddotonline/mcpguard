"""Fine-tune a small encoder as an injection classifier.

    python evals/ml/train.py BASE_MODEL_DIR DATA_DIR OUT_DIR [--seed N] [--epochs N]

Trains on DATA_DIR/train.jsonl and reports on DATA_DIR/dev.jsonl after each
epoch. Never reads evals/public/heldout/.
"""

import argparse
import json
import math
import random
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--maxlen", type=int, default=384)
a = ap.parse_args()

random.seed(a.seed)
torch.manual_seed(a.seed)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForSequenceClassification.from_pretrained(
    a.base, num_labels=2, id2label={0: "SAFE", 1: "INJECTION"}, label2id={"SAFE": 0, "INJECTION": 1}
).to(dev)


def load(p):
    return [json.loads(l) for l in open(p)]


train, devset = load(f"{a.data}/train.jsonl"), load(f"{a.data}/dev.jsonl")
# Balance the sources a little: no source more than 25% of an epoch.
cap = int(0.25 * len(train))
by = {}
for r in train:
    by.setdefault(r["source"] + r["label"], []).append(r)


def epoch_rows():
    rows = []
    for v in by.values():
        random.shuffle(v)
        rows += v[:cap]
    random.shuffle(rows)
    return rows


def collate(rows):
    enc = tok([r["text"] for r in rows], truncation=True, max_length=a.maxlen, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor([int(r["label"] == "injection") for r in rows])
    return enc


@torch.no_grad()
def evaluate():
    model.eval()
    scores = []
    for b in DataLoader(devset, batch_size=16, collate_fn=collate):
        b = {k: v.to(dev) for k, v in b.items()}
        p = torch.softmax(model(**{k: v for k, v in b.items() if k != "labels"}).logits.float(), -1)[:, 1]
        scores += p.tolist()
    model.train()
    ben = sorted((s for s, r in zip(scores, devset) if r["label"] != "injection"), reverse=True)
    thr = ben[max(0, math.ceil(0.01 * len(ben)) - 1)]  # 1% FPR on dev benign
    att = [s for s, r in zip(scores, devset) if r["label"] == "injection"]
    rec = sum(s > thr for s in att) / len(att)
    rec5 = sum(s >= 0.5 for s in att) / len(att)
    fpr5 = sum(s >= 0.5 for s in ben) / len(ben)
    per = {}
    for s, r in zip(scores, devset):
        k = (r["source"], r["label"])
        per.setdefault(k, []).append(s >= 0.5)
    return thr, rec, rec5, fpr5, {f"{k[0]}/{k[1][:3]}": round(sum(v) / len(v), 3) for k, v in sorted(per.items())}


steps = a.epochs * math.ceil(len(epoch_rows()) / a.batch)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
model.train()
t0 = time.time()
for ep in range(a.epochs):
    for i, b in enumerate(DataLoader(epoch_rows(), batch_size=a.batch, collate_fn=collate)):
        b = {k: v.to(dev) for k, v in b.items()}
        loss = model(**b).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if i % 200 == 0:
            print(f"epoch {ep} step {i} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    thr, rec, rec5, fpr5, per = evaluate()
    print(f"epoch {ep}: dev recall@1%FPR {rec:.3f} (thr {thr:.3f}); at 0.5: recall {rec5:.3f} FPR {fpr5:.4f}", flush=True)
    print("  per source flagged@0.5:", per, flush=True)
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
