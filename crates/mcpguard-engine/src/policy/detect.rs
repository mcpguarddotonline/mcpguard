//! Policy-level detectors: PII, custom regexes, and "never let this leave"
//! dictionaries stored only as salted hashes.

use std::collections::HashSet;

use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::Verdict;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Detectors {
    /// PII kinds to redact from content: email, phone, card, ssn, iban, all.
    #[serde(default)]
    pub pii: Vec<String>,
    #[serde(default)]
    pub custom: Vec<CustomDetector>,
    /// Terms that must never leave (in tool arguments), as
    /// `sha256(salt + lowercase term)` hex, so the policy file itself doesn't
    /// reveal them. Generate with `mcpguard policy hash-term`.
    #[serde(default)]
    pub never_leave: Option<NeverLeave>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomDetector {
    pub name: String,
    pub pattern: String,
    /// `redact` (replace the match), `ask` or `block` (the whole call).
    #[serde(default = "redact")]
    pub action: String,
    /// `arguments`, `results` or `both`.
    #[serde(default = "both")]
    pub applies_to: String,
}

fn redact() -> String {
    "redact".into()
}
fn both() -> String {
    "both".into()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeverLeave {
    pub salt: String,
    pub hashes: Vec<String>,
    /// Longest term in words (how many consecutive words to try).
    #[serde(default = "three")]
    pub max_words: usize,
}

fn three() -> usize {
    3
}

pub fn hash_term(salt: &str, term: &str) -> String {
    let d = Sha256::digest(format!("{salt}{}", term.trim().to_lowercase()).as_bytes());
    d.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Hit {
    pub detector: String,
    pub action: Verdict,
    pub redact: bool,
}

impl Detectors {
    pub fn validate(&self) -> Result<(), String> {
        for d in &self.custom {
            Regex::new(&d.pattern).map_err(|e| format!("detector '{}': {e}", d.name))?;
            if !["redact", "ask", "block"].contains(&d.action.as_str()) {
                return Err(format!("detector '{}': action must be redact, ask or block", d.name));
            }
            if !["arguments", "results", "both"].contains(&d.applies_to.as_str()) {
                return Err(format!(
                    "detector '{}': applies_to must be arguments, results or both",
                    d.name
                ));
            }
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.pii.is_empty() && self.custom.is_empty() && self.never_leave.is_none()
    }

    /// Scans one string. `outbound` = tool arguments (vs. results). Rewrites
    /// the string for redactions; returns what matched.
    pub fn apply(&self, s: &mut String, outbound: bool) -> Vec<Hit> {
        let mut hits = Vec::new();
        if !self.pii.is_empty() {
            let found = crate::pii::find(s, &self.pii);
            if !found.is_empty() {
                let kinds: HashSet<&str> = found.iter().map(|h| h.kind).collect();
                *s = crate::pii::redact(s, &found);
                for k in kinds {
                    hits.push(Hit {
                        detector: format!("pii.{k}"),
                        action: Verdict::Allow,
                        redact: true,
                    });
                }
            }
        }
        for d in &self.custom {
            let applies = d.applies_to == "both"
                || (outbound && d.applies_to == "arguments")
                || (!outbound && d.applies_to == "results");
            let Ok(re) = Regex::new(&d.pattern) else { continue };
            if !applies || !re.is_match(s) {
                continue;
            }
            match d.action.as_str() {
                "redact" => {
                    *s = re.replace_all(s, format!("«{}»", d.name).as_str()).into_owned();
                    hits.push(Hit {
                        detector: format!("custom.{}", d.name),
                        action: Verdict::Allow,
                        redact: true,
                    });
                }
                "ask" => hits.push(Hit {
                    detector: format!("custom.{}", d.name),
                    action: Verdict::Ask,
                    redact: false,
                }),
                _ => hits.push(Hit {
                    detector: format!("custom.{}", d.name),
                    action: Verdict::Block,
                    redact: false,
                }),
            }
        }
        if let (Some(nl), true) = (&self.never_leave, outbound) {
            let set: HashSet<&str> = nl.hashes.iter().map(String::as_str).collect();
            let words: Vec<&str> = s
                .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                .filter(|w| !w.is_empty())
                .collect();
            'outer: for n in 1..=nl.max_words.clamp(1, 6) {
                for w in words.windows(n) {
                    if set.contains(hash_term(&nl.salt, &w.join(" ")).as_str()) {
                        hits.push(Hit {
                            detector: "never_leave".into(),
                            action: Verdict::Block,
                            redact: false,
                        });
                        break 'outer;
                    }
                }
            }
        }
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_and_never_leave() {
        let d = Detectors {
            pii: vec!["email".into()],
            custom: vec![
                CustomDetector {
                    name: "ticket".into(),
                    pattern: r"ACME-\d{4}".into(),
                    action: "redact".into(),
                    applies_to: "both".into(),
                },
                CustomDetector {
                    name: "codename".into(),
                    pattern: r"(?i)project\s+falcon".into(),
                    action: "block".into(),
                    applies_to: "arguments".into(),
                },
            ],
            never_leave: Some(NeverLeave {
                salt: "s".into(),
                hashes: vec![hash_term("s", "Blue Heron")],
                max_words: 3,
            }),
        };
        assert!(d.validate().is_ok());
        let mut a = "mail bob@x.io about ACME-1234".to_owned();
        let h = d.apply(&mut a, false);
        assert_eq!(a, "mail «PII:email» about «ticket»");
        assert_eq!(h.len(), 2);
        let mut b = "status of Project Falcon".to_owned();
        assert_eq!(d.apply(&mut b, true)[0].action, Verdict::Block);
        let mut c = "send the blue heron numbers".to_owned();
        assert_eq!(d.apply(&mut c, true)[0].detector, "never_leave");
        let mut r = "send the blue heron numbers".to_owned();
        assert!(d.apply(&mut r, false).is_empty(), "never_leave only guards what leaves");
    }
}
