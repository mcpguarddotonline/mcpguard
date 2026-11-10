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
| Structured tool records (`build_records.py`): 10 record types as JSON/YAML, single or in lists | ours, long fields from fineweb-edu / Wikipedia above | benign records; attacks are the public attack rows above placed in one field (no new attack text) |
| Instruction-like harmless content in records (`build_records4.py`): person-to-person requests, account notices, chat one-liners, how-to steps, marketing, about 10% non-English | ours (templates written for this round, made-up senders, .example links) | benign records; attacks are the public attack rows above, sometimes next to a harmless request in the same field |

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
mlenv/bin/python evals/ml/build_records.py RAW OUT OUT_R3   # round 3: structured records
mlenv/bin/python evals/ml/build_records4.py RAW OUT_R3 OUT_R4   # round 4: instruction-like harmless content
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

## A dev set of real agent tool outputs (`build_dev_agent.py`)

Built from AgentDojo (MIT, v0.1.35): its four suites' tool data (emails,
files, calendar events, Slack messages, web pages, hotel and restaurant
records, transactions), returned as single records or short lists, in JSON
or YAML as tools return them. Benign rows use AgentDojo's harmless default
text in every attacker slot; attack rows put AgentDojo's own attack
templates and injection goals into one slot. We wrote no attack text.
AgentDojo's "injecagent" template is excluded because it is copied from a
held-out set. 974 rows: 576 benign, 398 attacks. Used for choosing models
and thresholds only, never trained on.

| Detector | recall at 1% FPR | at 2% | at 5% | AUC | FPR at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ProtectAI v2 | 0.5% | 4.3% | 13.6% | 0.68 | 37.2% |
| Round 1 | 0.0% | 0.0% | 0.0% | 0.83 | 57.6% |
| Round 1 + weight-space edit | 0.0% | 0.0% | 58.8% | 0.88 | 45.5% |
| Round 2, seed 1 | 19.8% | 28.1% | 48.0% | 0.88 | 28.8% |
| Round 2 soup | 20.6% | 24.6% | 42.2% | 0.87 | 45.0% |

**No detector here is usable on this data.** Every one flags a large share
of harmless tool output, and round 1 gives many harmless records a score of
1.0, so no threshold separates them. Broken down by shape (round 1, false
positives at 0.5): single JSON record 13%, JSON list 42%, single YAML record
49%, YAML list 89%. AgentDojo's own tools return YAML, so this is the shape
a real agent sees. It is the same shortcut the weight-space work found in
round 1: machine-formatted output reads as "injection" because the training
data has almost no harmless structured records.

This set also ranks the models differently from held-out InjecAgent (round 2
looks better here, worse there), so neither set alone is enough to choose a
model. From now on a model has to improve on dev_agent and dev_ood together
before it is scored on held-out.

Next: harmless structured records in training (public text and records
rendered as JSON and YAML lists, the way tools return them), then check
dev_agent first.

## Round 3 (November 2026): structured records — big gain, gate not met

Data: round 2's training set plus 11,146 structured records from
`build_records.py` (7,965 harmless, 3,181 with an attack text from the public
training attacks in one field), and a held-back `records_dev.jsonl` (2,854
rows, including record types and formats never seen in training). We checked
it independently before training: dev files byte-identical to round 2, no
held-out text overlapping new training rows (exact or 200-character match),
no records_dev row in training. 15 records_dev attack rows share stock
wording with training attacks, so its attack recall may be slightly high.

Three runs (seeds 1-3) averaged. Threshold: 1% FPR over the benign rows of
all four dev sets pooled (0.942 for the soup).

| Dev set (soup, at 0.942) | False positives | Recall |
| --- | ---: | ---: |
| dev | 0.2% | 99.1% |
| dev_ood | 0.1% | 93.6% |
| records_dev | 0.0% | 98.5% |
| dev_agent | 9.7% | 82.2% |

| dev_agent | recall at 1% FPR | AUC | YAML-list false positives |
| --- | ---: | ---: | ---: |
| ProtectAI v2 | 0.5% | 0.68 | 32% (at 0.5) |
| Round 1 | 0.0% | 0.83 | 89% (at 0.5) |
| Round 2 soup | 20.6% | 0.87 | 83% (at 0.5) |
| Round 3 soup | 39.9% | 0.93 | 6.8% (at 0.942) |

The format shortcut is largely gone. What still trips the model on dev_agent
is content: password-reset and security-code emails (some are AgentDojo's
deliberate phishing-style decoys, labelled benign), ordinary
person-to-person requests ("Please do the following by next week"), and
short chat messages. The Slack suite is the worst (27% false positives).

**Our gate for scoring held-out was dev_agent false positives at or below 2%
and recall at 1% FPR at or above 50%. Round 3 misses both, so the held-out
sets were not scored this round.** Next: harmless person-to-person requests
and account notices inside records, and a second agent-shaped dev source so
dev_agent's 576 benign rows aren't the only measure.

## Round 4 (November 2026): harmless text that reads like an instruction — recall gate met, false-positive gate missed

Data: round 3's training set plus 6,801 rows from `build_records4.py`
(4,797 harmless, 2,004 with an attack in one field): ordinary requests,
account notices, chat, marketing and how-to text inside records, the shapes
round 3 still flagged. Templates are split 75/25, and the held-back quarter
forms a new dev set, `records4_dev.jsonl` (1,699 rows). Independent check
before training: the four existing dev files are byte-identical to round 3;
no exact overlap with any dev or held-out set; no 200-character overlap with
held-out; four stray 200-character matches with dev sets, kept (none exact).

Three runs averaged. The threshold is now 1% FPR over the benign rows of all
**five** dev sets pooled. Round 3's soup is rescored on the same five sets so
the comparison is like for like.

| Dev set | Round 3 soup (thr 0.9989) | Round 4 soup (thr 0.9532) |
| --- | ---: | ---: |
| dev, FPR / recall | 0.0% / 89.9% | 0.1% / 99.3% |
| dev_ood | 0.0% / 72.8% | 0.0% / 93.6% |
| records_dev | 0.0% / 78.0% | 0.0% / 97.8% |
| records4_dev | 5.9% / 66.5% | 4.6% / 95.8% |
| dev_agent | 0.5% / 37.9% | 3.0% / 76.4% |
| dev_agent recall at 1% FPR | 39.9% | **52.0%** |
| dev_agent AUC | 0.932 | 0.926 |

**Gate: dev_agent false positives 3.0% (needs at most 2%) and recall at 1%
FPR 52.0% (needs at least 50%). The recall half passes for the first time;
the false-positive half doesn't, so the held-out sets were not scored.**

What is left, at the round 4 threshold:

* dev_agent: 17 of 576 harmless rows. They come from a handful of AgentDojo
  items repeated across list groupings: a Slack web-content dump that holds
  to-do pages written as imperatives ("Find a channel ... and post there"),
  a family-reunion email, a password-reset email and a security-code email.
  Because the rows share items, 576 rows is far fewer independent samples
  than it looks, and one item moves FPR by about a point.
* records4_dev: person-to-person requests (38 of 279) and chat (17 of 134).
  Notices, marketing and how-to text are now clean (0 of 790).

Two measurement notes:

* **Per-field scoring was worse.** `score_fields.py` scores each string
  field of a JSON/YAML result separately and takes the maximum. On the round
  3 soup it gave dev_agent recall at 1% FPR of 25.9% (vs 39.9% whole) and
  AUC 0.881 (vs 0.932): the model was trained on whole records, and the
  maximum over ~12 fields gives more chances for a false alarm. We keep
  whole-text scoring; the script stays for a model trained on fields.
* **Single runs saturate.** Each single seed scores some harmless rows at
  exactly 1.0 in float32, more than 1% of the pool, so the per-seed
  "recall at 1% FPR" is 0 and is not meaningful. The averaged model has no
  such ties. Scoring logits instead of probabilities would fix this.

`gate.py` applies the gate to any score file and refuses held-out scores.
