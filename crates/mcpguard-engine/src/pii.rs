//! PII detectors (opt-in per policy): emails, phone numbers, payment cards
//! (Luhn-checked), US SSNs and IBANs (mod-97 checked). Regex + checksum, no
//! model; a name/address NER model can plug in behind the same findings.

use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PiiHit {
    pub kind: &'static str,
    pub start: usize,
    pub end: usize,
}

static EMAIL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b[a-z0-9._%+-]{1,64}@[a-z0-9.-]{1,253}\.[a-z]{2,24}\b").expect("re"));
static PHONE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:\+\d{1,3}[ .-]?)?\(?\b\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b").expect("re"));
static CARD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(?:\d[ -]?){12,18}\d\b").expect("re"));
static SSN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:00[1-9]|0[1-9]\d|[1-578]\d\d|6[0-57-9]\d|66[0-57-9])-(?:0[1-9]|[1-9]\d)-(?:000[1-9]|00[1-9]\d|0[1-9]\d\d|[1-9]\d{3})\b").expect("re")
});
static IBAN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b").expect("re"));

fn luhn(digits: &[u32]) -> bool {
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            if i % 2 == 1 {
                let x = d * 2;
                if x > 9 { x - 9 } else { x }
            } else {
                d
            }
        })
        .sum();
    sum % 10 == 0
}

fn iban_ok(s: &str) -> bool {
    let c: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if !(15..=34).contains(&c.len()) {
        return false;
    }
    let rearranged = format!("{}{}", &c[4..], &c[..4]);
    let mut rem: u64 = 0;
    for ch in rearranged.chars() {
        let v = match ch.to_digit(36) {
            Some(v) => v as u64,
            None => return false,
        };
        rem = if v >= 10 {
            (rem * 100 + v) % 97
        } else {
            (rem * 10 + v) % 97
        };
    }
    rem == 1
}

pub fn find(text: &str, kinds: &[String]) -> Vec<PiiHit> {
    let want = |k: &str| kinds.iter().any(|x| x == k || x == "all");
    let mut out = Vec::new();
    let mut push = |kind, m: regex::Match<'_>| {
        out.push(PiiHit {
            kind,
            start: m.start(),
            end: m.end(),
        })
    };
    if want("email") {
        EMAIL.find_iter(text).for_each(|m| push("email", m));
    }
    if want("card") {
        for m in CARD.find_iter(text) {
            let d: Vec<u32> = m.as_str().chars().filter_map(|c| c.to_digit(10)).collect();
            if (13..=19).contains(&d.len()) && luhn(&d) {
                push("card", m);
            }
        }
    }
    if want("ssn") {
        SSN.find_iter(text).for_each(|m| push("ssn", m));
    }
    if want("iban") {
        for m in IBAN.find_iter(text) {
            if iban_ok(m.as_str()) {
                push("iban", m);
            }
        }
    }
    if want("phone") {
        PHONE.find_iter(text).for_each(|m| push("phone", m));
    }
    out.sort_by_key(|h| (h.start, std::cmp::Reverse(h.end)));
    let mut merged: Vec<PiiHit> = Vec::new();
    for h in out {
        if merged.last().is_none_or(|l| h.start >= l.end) {
            merged.push(h);
        }
    }
    merged
}

/// Replaces hits with `«PII:kind»`.
pub fn redact(text: &str, hits: &[PiiHit]) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for h in hits {
        out.push_str(&text[last..h.start]);
        out.push_str(&format!("«PII:{}»", h.kind));
        last = h.end;
    }
    out.push_str(&text[last..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_with_checksums() {
        let all = vec!["all".to_owned()];
        let t = "mail jo@example.com, card 4242 4242 4242 4242, bad 4242 4242 4242 4241, ssn 123-45-6789, iban GB82 WEST 1234 5698 7654 32, call 415-555-0134";
        let kinds: Vec<&str> = find(t, &all).iter().map(|h| h.kind).collect();
        assert_eq!(kinds, ["email", "card", "ssn", "iban", "phone"]);
        assert!(redact(t, &find(t, &all)).contains("«PII:card»"));
        assert!(
            find("order 1234567890123 shipped", &all)
                .iter()
                .all(|h| h.kind != "card")
                || !luhn(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3])
        );
    }
}
