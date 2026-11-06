//! Secret detection: high-precision, vendor-specific patterns.
//!
//! Every rule keys on a vendor prefix or a fixed structure so false positives
//! stay rare. When adding a rule, add one positive and one negative test.

use std::sync::LazyLock;

use regex::Regex;

use crate::verdict::{Action, Finding, Kind};

struct SecretRule {
    id: &'static str,
    label: &'static str,
    re: Regex,
    /// Capture group holding the secret itself (0 = whole match).
    group: usize,
    /// Extra check that must pass for a match to count.
    validate: Option<fn(&str) -> bool>,
}

fn rule(id: &'static str, label: &'static str, pattern: &str, group: usize) -> SecretRule {
    SecretRule {
        id,
        label,
        re: Regex::new(pattern).expect("built-in secret rule must compile"),
        group,
        validate: None,
    }
}

static RULES: LazyLock<Vec<SecretRule>> = LazyLock::new(|| {
    vec![
        rule(
            "secret.aws_access_key_id",
            "aws_access_key",
            r"\b((?:AKIA|ASIA)[0-9A-Z]{16})\b",
            1,
        ),
        rule(
            "secret.github_token",
            "github_token",
            r"\b((?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36})\b",
            1,
        ),
        rule(
            "secret.github_fine_grained_pat",
            "github_token",
            r"\b(github_pat_[A-Za-z0-9_]{82})\b",
            1,
        ),
        rule(
            "secret.stripe_live_key",
            "stripe_live_key",
            r"\b((?:sk|rk)_live_[A-Za-z0-9]{24,})\b",
            1,
        ),
        rule(
            "secret.slack_token",
            "slack_token",
            r"\b(xox[abprs]-[A-Za-z0-9-]{10,})\b",
            1,
        ),
        rule(
            "secret.anthropic_api_key",
            "anthropic_key",
            r"\b(sk-ant-[A-Za-z0-9_-]{32,})",
            1,
        ),
        SecretRule {
            validate: Some(high_entropy),
            ..rule(
                "secret.openai_api_key",
                "openai_key",
                r"\b(sk-(?:proj-)?[A-Za-z0-9_-]{40,})",
                1,
            )
        },
        rule(
            "secret.private_key_pem",
            "private_key",
            r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]+?-----END (?:[A-Z]+ )?PRIVATE KEY-----",
            0,
        ),
        rule(
            "secret.db_url_password",
            "db_password",
            r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^:\s/@]+:([^@\s/]{6,})@",
            1,
        ),
    ]
});

/// Rejects low-entropy look-alikes such as `sk-aaaaaaaa…`.
fn high_entropy(s: &str) -> bool {
    shannon(s) >= 3.5
}

pub(crate) fn shannon(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut freq = [0usize; 256];
    for b in s.bytes() {
        freq[b as usize] += 1;
    }
    let n = s.len() as f64;
    freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

pub(crate) fn find_secrets(text: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    for r in RULES.iter() {
        for caps in r.re.captures_iter(text) {
            let Some(m) = caps.get(r.group) else { continue };
            if r.validate.is_some_and(|ok| !ok(m.as_str())) {
                continue;
            }
            out.push(Finding {
                rule_id: r.id,
                kind: Kind::Secret,
                action: Action::Redact,
                start: m.start(),
                end: m.end(),
                label: Some(r.label),
                note: "credential detected",
            });
        }
    }
    out
}
