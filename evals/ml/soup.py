"""Average the weights of several fine-tuned runs ("model soup").

    python evals/ml/soup.py OUT_DIR CKPT1 CKPT2 [CKPT3 ...]

All runs must start from the same base model, so their weights stay close
enough that the average is itself a good model (Wortsman et al., 2022).
"""

import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

out, ckpts = sys.argv[1], sys.argv[2:]
models = [AutoModelForSequenceClassification.from_pretrained(c) for c in ckpts]
sds = [m.state_dict() for m in models]
avg = {}
for k, v in sds[0].items():
    avg[k] = (sum(sd[k].float() for sd in sds) / len(sds)).to(v.dtype) if v.is_floating_point() else v
models[0].load_state_dict(avg)
models[0].save_pretrained(out)
AutoTokenizer.from_pretrained(ckpts[0]).save_pretrained(out)
print(f"averaged {len(ckpts)} runs into {out}")
