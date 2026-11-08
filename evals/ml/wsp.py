"""Weight-space programming harness: inspect and hand-edit a trained classifier.

No gradients and no training loops. Every forward pass goes through
`Session.forward`, which counts against a budget and only accepts dev or
train texts (never held-out). Each edit is logged with how many weights it
changed, so the final diff can be published.

    from wsp import Session
    s = Session(CKPT, DATA_DIR, budget=50)
"""

import json
import math

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Session:
    def __init__(self, ckpt, data_dir, budget=50, maxlen=384):
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(self.dev).eval()
        self.orig = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.budget, self.used, self.maxlen = budget, 0, maxlen
        self.rows = {s: [json.loads(l) for l in open(f"{data_dir}/{s}.jsonl")] for s in ("train", "dev")}
        self.log = []

    # -------------------------------------------------------------- reading
    def params(self):
        return {k: tuple(v.shape) for k, v in self.model.named_parameters()}

    def weight(self, name):
        return dict(self.model.named_parameters())[name].detach()

    # ------------------------------------------------------- forward passes
    @torch.no_grad()
    def forward(self, texts, hidden=False, batch=16):
        """One budgeted forward pass over up to 256 texts. Returns injection
        probabilities and, if asked, the per-layer [CLS] hidden states."""
        assert len(texts) <= 256, "one forward pass is at most 256 texts"
        assert self.used < self.budget, "forward-pass budget spent"
        self.used += 1
        probs, hs = [], []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i : i + batch], truncation=True, max_length=self.maxlen, padding=True, return_tensors="pt").to(self.dev)
            out = self.model(**enc, output_hidden_states=hidden)
            probs += torch.softmax(out.logits.float(), -1)[:, 1].tolist()
            if hidden:
                hs.append(torch.stack([h[:, 0].float().cpu() for h in out.hidden_states], 1))
        return (probs, torch.cat(hs)) if hidden else probs

    # --------------------------------------------------------------- writing
    @torch.no_grad()
    def edit(self, name, new_value, why):
        p = dict(self.model.named_parameters())[name]
        changed = int((new_value.to(p.device) != p).sum())
        p.copy_(new_value.to(p.device, p.dtype))
        self.log.append({"param": name, "changed": changed, "why": why})
        return changed

    def total_changed(self):
        sd = self.model.state_dict()
        n = sum(int((sd[k] != v.to(sd[k].device)).sum()) for k, v in self.orig.items())
        tot = sum(v.numel() for v in self.orig.values())
        return n, tot, n / tot

    @torch.no_grad()
    def revert(self):
        self.model.load_state_dict(self.orig)
        self.log.append({"revert": True})

    def save(self, out):
        self.model.save_pretrained(out)
        self.tok.save_pretrained(out)
        n, tot, frac = self.total_changed()
        json.dump({"edits": self.log, "weights_changed": n, "weights_total": tot, "fraction": frac,
                   "forward_passes_used": self.used}, open(f"{out}/wsp_log.json", "w"), indent=1)


def recall_at_fpr(scores, labels, fpr=0.01):
    ben = sorted((s for s, l in zip(scores, labels) if l != "injection"), reverse=True)
    thr = ben[max(0, math.ceil(fpr * len(ben)) - 1)]
    att = [s for s, l in zip(scores, labels) if l == "injection"]
    return sum(s > thr for s in att) / len(att), thr


def features(session, texts):
    """Budgeted pass that returns the classifier's input vectors (after the
    head) and the injection probabilities."""
    got = []
    h = session.model.classifier.register_forward_hook(lambda m, i, o: got.append(i[0].detach().float().cpu()))
    try:
        p = session.forward(texts)
    finally:
        h.remove()
    return torch.cat(got), p
