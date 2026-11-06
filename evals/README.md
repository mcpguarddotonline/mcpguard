# MCP-Guard detection evals

`injection.jsonl` is a small **seed** corpus (20 injection attempts, 25 benign
texts that look instruction-like: READMEs, issue text, recipes, non-Latin
scripts, emoji). Run it with:

```bash
cargo test -p mcpguard-engine --test eval -- --nocapture
```

Current rule-based layer on the seed set: recall 1.00, precision 0.95,
false-positive rate 0.04.

**Read these numbers with care.** The seed set was written by the same people
who wrote the rules, so it overstates real-world recall. Before we publish
headline numbers, the suite will add public third-party datasets (for example
deepset/prompt-injections, BIPIA-style indirect injections in emails and web
pages, and tool-output corpora) and the ML classifier layer (an ONNX model
behind the same `Finding` type). Contributions of real-world samples, both
attacks and false positives, are very welcome: add a JSON line with `text` and
`label` (`injection` or `benign`).
