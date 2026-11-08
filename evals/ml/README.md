# Injection classifier: training and evaluation

A small encoder fine-tuned to spot prompt injection in what tools return. It
sits behind the same `Finding` type as the rules and is measured on the same
held-out sets in [`../README.md`](../README.md).

Everything here runs on a laptop (we use an Apple M4, 16 GB) at no cost.

## Rules we hold ourselves to

1. **Held-out stays unseen.** Nothing in `evals/public/heldout/` is used for
   training, threshold choice or weight edits. Thresholds are chosen on our
   dev split (1% false-positive rate on dev benign texts) and frozen before the
   held-out run.
2. **Licensed data only**, listed below. Datasets with no stated license are
   left out even when they look useful.
3. **Same windows for everyone.** Every detector, ours and the baselines, sees
   texts in the same overlapping 512-token windows and keeps its highest
   score.

## Data

| Source | License | Used as |
| --- | --- | --- |
| deepset/prompt-injections (train) | Apache-2.0 | attacks and benign |
| Lakera/gandalf_ignore_instructions (train) | MIT | attacks |
| microsoft/llmail-inject-challenge (phase 2) | MIT | attacker emails aimed at an email agent; benign emails |
| jackhhao/jailbreak-classification (train) | Apache-2.0 | attacks and benign |
| reshabhs/SPML_Chatbot_Prompt_Injection | MIT | attacks and benign |
| yanismiraoui/prompt_injections | Apache-2.0 | attacks |
| HuggingFaceFW/fineweb-edu (4,000 pages) | ODC-By | benign web text |
| wikimedia/wikipedia (2,000 articles) | CC-BY-SA | benign prose |
| Local crate READMEs, Rust source, man pages | various open source | benign file and shell output |
| Synthetic tool outputs (`build_data.py`) | ours | attacks and benign |

Left out on purpose: Lakera/mosscap (many entries are ordinary chat with a
password game, not injections, and it's the same game family as our Gandalf
held-out set, so it would flatter that number).

**About the synthetic set.** `build_data.py` wraps benign text in tool-output
shapes (emails, reviews, issues, web pages, chats, calendar events, search
results) and, for attacks, inserts one instruction built from generic attacker
goals and framings. Benign emails and chats also get ordinary person-to-person
requests ("Could you send me the slides?") so the model learns that a request
alone isn't an attack. Disclosure: we wrote these templates ourselves and had
seen a few InjecAgent examples while working on the rules, so InjecAgent
results should be read with that in mind. We did not copy InjecAgent text,
goals or tool names.

## Reproduce

```bash
python -m venv mlenv && mlenv/bin/pip install torch transformers safetensors huggingface_hub pandas
# downloads: see the commands at the top of build_data.py
mlenv/bin/python evals/ml/build_data.py RAW OUT
mlenv/bin/python evals/ml/train.py BASE_MODEL OUT CKPT --seed 1
mlenv/bin/python evals/ml/score.py CKPT scores.json evals/public/heldout/*.jsonl
mlenv/bin/python evals/ml/metrics.py THRESHOLD ours=scores.json
```

## Results (November 2026, first run)

Base model: Ettin-32M (MIT), fine-tuned for 3 epochs on the data above, about
an hour on an M4. Baseline: ProtectAI deberta-v3-base-prompt-injection-v2
(Apache-2.0, 184M), the strongest openly downloadable injection classifier.
Meta's PromptGuard 2 is not included yet: it needs its license accepted on
Hugging Face.

At each model's default threshold (0.5), held-out sets:

| Detector | deepset recall | Gandalf recall | InjecAgent recall | InjecAgent false positives |
| --- | ---: | ---: | ---: | ---: |
| Rules only (engine today) | 23.3% | 62.8% | 1.6% | 0.0% |
| ProtectAI v2 (184M) | 36.7% | 100.0% | 65.7% | **72.0%** |
| Ours, fine-tuned (32M) | **86.7%** | 100.0% | 69.1% | 9.3% |
| Ours + weight-space edit | 80.0% | 99.6% | 59.8% | 3.7% |

At the same false-positive rate on InjecAgent's 2,214 benign tool outputs
(a diagnostic comparison; the threshold here is read off the held-out set,
so these are not shippable operating points):

| Detector | recall at 1% FPR | at 2% | at 3.7% |
| --- | ---: | ---: | ---: |
| ProtectAI v2 | 6.5% | 7.8% | 8.9% |
| Ours, fine-tuned | 34.3% | 51.0% | 61.3% |
| Ours + weight-space edit | 36.9% | 50.8% | 59.8% |

### The weight-space edit, honestly

On a second dev set of tool outputs in formats the training data never used
(`build_ood.py`), the fine-tuned model flagged 42% of harmless log excerpts and
21% of harmless Python-style email dicts. Reading per-token scores showed why:
log-level labels (`WARN`, `INFO`, `worker:`) and `'key':` quoting pushed toward
"injection", a shortcut learned from attacks that fake system markers. We
found that direction in the final token states, removed its overlap with the
real attack direction (on which attacks and normal text then score the same),
and projected it out of the head's input matrix: a rank-1 edit, no gradients,
2 forward passes to reproduce (`wsp_edit.py`).

On unseen dev data it cut false positives on new formats from 6.1% to 2.0% at
almost no recall cost. On the held-out sets it mostly behaves like a higher
threshold: at equal false-positive rates it is within about two points of the
plain fine-tune either way. So it is **not** a clear improvement and we don't
claim one. The finding is still useful: it says exactly which benign data the
training set lacks (machine-formatted output), which the next run adds.

Order of operations, for transparency: the plain fine-tune was scored on the
held-out sets once before the edit was designed; the edit was designed only
from dev data and scored on held-out once afterwards.
