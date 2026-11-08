//! Scores the injection rules against JSONL corpora.
//!
//!     cargo run --release -p mcpguard-engine --example eval -- evals/public/*.jsonl
//!
//! Each line: {"text": "...", "label": "injection" | "benign"}. A text counts
//! as flagged when the rules report an injection finding (secret redaction
//! alone doesn't count: that's a different shield). Add `--misses N` to print
//! up to N missed injections and N false positives per file.

use std::collections::BTreeMap;

use mcpguard_engine::{Direction, Kind, inspect};

#[derive(Default)]
struct Score {
    tp: usize,
    fp: usize,
    fn_: usize,
    tn: usize,
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut show = 0usize;
    if let Some(i) = args.iter().position(|a| a == "--misses") {
        show = args.get(i + 1).and_then(|n| n.parse().ok()).unwrap_or(5);
        args.drain(i..(i + 2).min(args.len()));
    }
    if args.is_empty() {
        eprintln!("usage: eval [--misses N] FILE.jsonl...");
        std::process::exit(2);
    }
    println!("| Corpus | Injections | Benign | Recall | False-positive rate | Precision |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: |");
    let mut total = Score::default();
    let mut details = Vec::new();
    for path in &args {
        let data = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut s = Score::default();
        let mut missed = Vec::new();
        let mut false_pos: Vec<(String, String)> = Vec::new();
        let mut rules: BTreeMap<&'static str, usize> = BTreeMap::new();
        for line in data.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(row) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let (Some(text), Some(label)) = (row["text"].as_str(), row["label"].as_str()) else {
                continue;
            };
            let v = inspect(text, Direction::ToClient);
            let hits: Vec<&'static str> = v
                .findings
                .iter()
                .filter(|f| f.kind == Kind::Injection)
                .map(|f| f.rule_id)
                .collect();
            let flagged = !hits.is_empty();
            for r in &hits {
                *rules.entry(r).or_default() += 1;
            }
            match (label == "injection", flagged) {
                (true, true) => s.tp += 1,
                (true, false) => {
                    s.fn_ += 1;
                    missed.push(text.to_owned());
                }
                (false, true) => {
                    s.fp += 1;
                    false_pos.push((hits.join(","), text.to_owned()));
                }
                (false, false) => s.tn += 1,
            }
        }
        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or(path);
        println!("{}", row(name, &s));
        total.tp += s.tp;
        total.fp += s.fp;
        total.fn_ += s.fn_;
        total.tn += s.tn;
        if show > 0 {
            details.push(format!("\n## {name}\nrules fired: {rules:?}"));
            for t in missed.iter().take(show) {
                details.push(format!("  missed: {}", clip(t)));
            }
            for (r, t) in false_pos.iter().take(show) {
                details.push(format!("  false positive [{r}]: {}", clip(t)));
            }
        }
    }
    if args.len() > 1 {
        println!("{}", row("**All**", &total));
    }
    for d in details {
        println!("{d}");
    }
}

fn row(name: &str, s: &Score) -> String {
    let pct = |n: usize, d: usize| {
        if d == 0 {
            "–".to_owned()
        } else {
            format!("{:.1}%", 100.0 * n as f64 / d as f64)
        }
    };
    format!(
        "| {name} | {} | {} | {} | {} | {} |",
        s.tp + s.fn_,
        s.fp + s.tn,
        pct(s.tp, s.tp + s.fn_),
        pct(s.fp, s.fp + s.tn),
        pct(s.tp, s.tp + s.fp),
    )
}

fn clip(t: &str) -> String {
    let one: String = t.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() > 140 {
        format!("{}…", one.chars().take(140).collect::<String>())
    } else {
        one
    }
}
