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

## Round 2 (November 2026): more public data and a model soup — a regression

Changes (`add_round2.py`, `soup.py`): 5,684 LLMail-Inject phase 2 attack
emails that round 1 did not use, 2,750 Wikipedia articles in 11 languages as
benign text, three training runs (seeds 1-3) averaged into one model. Texts
in dev or dev_ood were excluded from training. The threshold was chosen on
dev + dev_ood benign at 1% FPR (0.112) and frozen before one held-out run.

On dev data it looked better than round 1:

| Detector | dev_ood false positives | dev_ood recall |
| --- | ---: | ---: |
| Round 1 (dev-chosen threshold) | 2.1% | 93.0% |
| Round 2 single run (seed 1) | 1.9% | 95.4% |
| Round 2 soup of 3 runs | **1.4%** | **96.4%** |

On the held-out sets it is worse:

| Detector (dev-chosen threshold) | deepset recall | Gandalf recall | InjecAgent recall | InjecAgent FPR |
| --- | ---: | ---: | ---: | ---: |
| Round 1 (0.999) | 76.7% | 99.1% | 52.0% | 2.1% |
| Round 2 soup (0.112) | 88.3% | 100.0% | 50.2% | **12.3%** |

| InjecAgent recall at equal FPR | 1% | 2% | 3.7% |
| --- | ---: | ---: | ---: |
| ProtectAI v2 | 6.5% | 7.8% | 8.9% |
| Round 1 | **34.3%** | **51.0%** | **61.3%** |
| Round 2 soup | 22.2% | 26.9% | 33.5% |

So round 1 stays our best model. What we take from it:

* **Our dev sets don't predict tool-output performance.** Dev improved on
  every measure while held-out InjecAgent got clearly worse. The next step is
  a dev set made of agent tool outputs from a source other than InjecAgent,
  used only for choosing thresholds and models.
* More attacks from one source (emails aimed at one email agent) seem to have
  narrowed what the model treats as an attack in other tool outputs.
* The held-out sets have now been scored four times (round 1, the
  weight-space edit, round 2). Every run is reported here, but repeated use
  wears out a held-out set, so future model choices must be made on dev
  alone and the held-out sets kept for a final check.
