"""Score JSONL texts with a Hugging Face sequence classifier.

    python evals/ml/score.py MODEL_DIR OUT.json FILE.jsonl [FILE.jsonl ...]

Long texts are split into overlapping 512-token windows and the highest
injection probability is kept, the same way the engine will run it. Writes
{"file": [[label, score], ...]} so metrics.py can compare detectors.
"""

import json
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

WIN, STRIDE = 512, 384


def injection_index(model):
    for i, name in model.config.id2label.items():
        if str(name).upper() in ("INJECTION", "LABEL_1", "1", "MALICIOUS", "JAILBREAK"):
            return int(i)
    return 1


def windows(tok, text):
    ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    body = WIN - 2
    if len(ids) <= body:
        return [ids]
    return [ids[i : i + body] for i in range(0, max(1, len(ids) - body + STRIDE), STRIDE)]


@torch.no_grad()
def score_texts(model, tok, texts, device, batch=8):
    k = injection_index(model)
    flat, owner = [], []
    for n, t in enumerate(texts):
        for w in windows(tok, t):
            flat.append([tok.cls_token_id] + w + [tok.sep_token_id])
            owner.append(n)
    best = [0.0] * len(texts)
    order = sorted(range(len(flat)), key=lambda i: len(flat[i]))
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        L = max(len(flat[i]) for i in idx)
        pad = tok.pad_token_id or 0
        ids = torch.tensor([flat[i] + [pad] * (L - len(flat[i])) for i in idx], device=device)
        mask = torch.tensor([[1] * len(flat[i]) + [0] * (L - len(flat[i])) for i in idx], device=device)
        p = torch.softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)[:, k].tolist()
        for i, v in zip(idx, p):
            best[owner[i]] = max(best[owner[i]], v)
    return best


def main():
    model_dir, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    res = {}
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        t0 = time.time()
        s = score_texts(model, tok, [r["text"] for r in rows], device)
        print(f"{f}: {len(rows)} texts in {time.time() - t0:.1f}s", file=sys.stderr)
        res[f.split("/")[-1].removesuffix(".jsonl")] = [[r["label"], v] for r, v in zip(rows, s)]
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
