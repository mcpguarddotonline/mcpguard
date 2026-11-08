"""The weight-space edit we ship, reproducible from dev data alone.

    python evals/ml/wsp_edit.py CKPT DATA_DIR OUT_DIR

What it does (found by inspecting the fine-tuned model, see README):
harmless tool output in machine formats (log lines like `WARN worker:`,
Python-style `'subject':` dicts) pushed the classifier toward "injection"
through one direction in the final token states. We take that direction,
remove the part shared with real attack tokens, and project what's left out
of the head's input matrix. A rank-1 change to `head.dense.weight`; no
gradients, no training. Uses half A of dev_ood.jsonl and the dev split only.
"""

import json
import random
import sys

import torch

from wsp import Session

ckpt, data, out = sys.argv[1:4]
random.seed(0)
s = Session(ckpt, data, budget=50)
m = s.model
ood = [json.loads(l) for l in open(f"{data}/dev_ood.jsonl")]
half_a = ood[0::2]


@torch.no_grad()
def token_states(texts):
    s.used += 1
    H, L = [], []
    for t in texts:
        enc = s.tok(t, truncation=True, max_length=384, return_tensors="pt").to(s.dev)
        h = m.model(**enc).last_hidden_state[0]
        lg = m.classifier(m.head(h)).float()
        H.append(h.float().cpu())
        L.append((lg[:, 1] - lg[:, 0]).cpu())
    return torch.cat(H), torch.cat(L)


ben = [r["text"] for r in half_a if r["label"] == "benign" and r["source"] in ("ood_logs", "ood_mail")][:80]
att = [r["text"] for r in s.rows["dev"] if r["label"] == "injection" and r["source"] in ("llmail", "spml", "deepset", "jackhhao")]
random.shuffle(att)
hb, lb = token_states(ben)
ha, la = token_states(att[:80])
shortcut = hb[lb > 4].mean(0) - hb[lb < 0].mean(0)
attack = ha[la > 4].mean(0) - hb[lb < 0].mean(0)
r = shortcut - (shortcut @ attack) / (attack @ attack) * attack
r = r / r.norm()

W = s.weight("head.dense.weight").clone()
rr = r.to(W.device, W.dtype)
s.edit("head.dense.weight", W - (W @ rr)[:, None] * rr[None, :],
       "project out the machine-format shortcut direction (orthogonal to the attack direction) from the head input")
s.save(out)
torch.save(r, f"{out}/wsp_direction.pt")
n, tot, frac = s.total_changed()
print(f"edited {n} of {tot} weights ({frac:.2%}), rank-1, {s.used} forward passes")
