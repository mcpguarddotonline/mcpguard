"""Score tool output field by field, the way a gateway sees it.

    python evals/ml/score_fields.py MODEL_DIR OUT.json FILE.jsonl [FILE.jsonl ...]

score.py reads a tool result as one block of text, so several harmless
records in one list can add up to a false alarm. Here, if a text parses as
JSON or YAML into records, each string value of at least MIN_CHARS
characters is scored on its own (as "key: value", so the model still sees
what the field is) and the text's score is the highest field score. Text
that doesn't parse into records is scored whole, exactly as in score.py.

The engine receives tool results as JSON-RPC payloads, so it can do the
same split without guessing. Output format matches score.py.
Set DEVICE=cpu to run beside a training job.
"""

import json
import os
import sys
import time

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from score import score_texts

MIN_CHARS = 16


def parse(text):
    t = text.strip()
    if not t:
        return None
    try:
        v = json.loads(t)
    except ValueError:
        try:
            v = yaml.safe_load(t)
        except yaml.YAMLError:
            return None
    return v if isinstance(v, (dict, list)) else None


def fields(v, key=""):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from fields(x, str(k))
    elif isinstance(v, list):
        for x in v:
            yield from fields(x, key)
    elif isinstance(v, str) and len(v.strip()) >= MIN_CHARS:
        yield f"{key}: {v}" if key else v


def units(text):
    v = parse(text)
    fs = list(fields(v)) if v is not None else []
    return fs or [text]


def main():
    model_dir, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    device = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    res = {}
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        t0 = time.time()
        flat, owner = [], []
        for n, r in enumerate(rows):
            for u in units(r["text"]):
                flat.append(u)
                owner.append(n)
        s = score_texts(model, tok, flat, device)
        best = [0.0] * len(rows)
        for i, v in zip(owner, s):
            best[i] = max(best[i], v)
        print(f"{f}: {len(rows)} texts, {len(flat)} fields in {time.time() - t0:.1f}s", file=sys.stderr)
        res[f.split("/")[-1].removesuffix(".jsonl")] = [[r["label"], v] for r, v in zip(rows, best)]
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
