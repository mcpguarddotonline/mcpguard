//! Vault primitives shared by the hosted vault, the gateway and the CLI.
//!
//! - **Handles**: agents refer to secrets as `{{vault:name}}`, never by value.
//! - **Masking without plaintext**: a [`MaskSet`] finds exact occurrences of
//!   secret values (and their base64 / URL-encoded forms) in text and swaps
//!   them back to handles. It holds only per-value *verifiers* (length, a
//!   rolling hash, and an HMAC under a per-org key), so the component doing
//!   the masking never holds the secrets themselves. Verifiers still allow
//!   offline guessing of weak passwords if stolen; high-entropy API keys are
//!   safe.
//! - **Destinations**: hosts a tool call is headed to, for destination-bound
//!   grants.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use base64::Engine;
use hmac::{Hmac, Mac};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Sha256;

static HANDLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{\{\s*vault:([a-z0-9][a-z0-9_.-]{0,63})\s*\}\}").expect("handle regex"));

/// Secret names: lowercase letters, digits, `_`, `.`, `-`; 1–64 chars.
pub fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_.-".contains(&b))
        && name.as_bytes()[0].is_ascii_alphanumeric()
}

pub fn handle(name: &str) -> String {
    format!("{{{{vault:{name}}}}}")
}

/// One handle found in a JSON value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HandleUse {
    pub name: String,
    /// JSON path of the string, e.g. `headers.Authorization`.
    pub path: String,
}

fn walk_mut(v: &mut Value, path: String, f: &mut impl FnMut(&str, &mut String)) {
    match v {
        Value::String(s) => f(&path, s),
        Value::Array(a) => {
            for (i, x) in a.iter_mut().enumerate() {
                walk_mut(x, format!("{path}[{i}]"), f);
            }
        }
        Value::Object(m) => {
            for (k, x) in m.iter_mut() {
                let p = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                walk_mut(x, p, f);
            }
        }
        _ => {}
    }
}

/// Every handle in `v` (object keys are not searched).
pub fn find_handles(v: &Value) -> Vec<HandleUse> {
    let mut out = Vec::new();
    let mut v = v.clone();
    walk_mut(&mut v, String::new(), &mut |path, s| {
        for c in HANDLE.captures_iter(s) {
            out.push(HandleUse {
                name: c[1].to_owned(),
                path: path.to_owned(),
            });
        }
    });
    out
}

pub fn contains_handle(s: &str) -> bool {
    HANDLE.is_match(s)
}

/// Replaces handles in `v`. `resolve(name, path)` returns the value to put
/// in, or `None` to leave the handle untouched.
pub fn substitute(v: &mut Value, mut resolve: impl FnMut(&str, &str) -> Option<String>) {
    walk_mut(v, String::new(), &mut |path, s| {
        if !HANDLE.is_match(s) {
            return;
        }
        let out = HANDLE
            .replace_all(s, |c: &regex::Captures<'_>| {
                resolve(&c[1], path).unwrap_or_else(|| c[0].to_owned())
            })
            .into_owned();
        *s = out;
    });
}

/// Hosts a tool call is headed to: every URL-looking string in the
/// arguments, plus `host` / `hostname` / `domain` fields.
pub fn destinations(args: &Value) -> Vec<String> {
    static URL: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)\b(?:https?|wss?)://([a-z0-9.-]+)(?::\d+)?").expect("url regex"));
    let mut out = HashSet::new();
    let mut v = args.clone();
    walk_mut(&mut v, String::new(), &mut |path, s| {
        for c in URL.captures_iter(s) {
            out.insert(c[1].to_ascii_lowercase());
        }
        let key = path.rsplit('.').next().unwrap_or(path).to_ascii_lowercase();
        if matches!(key.as_str(), "host" | "hostname" | "domain") && !s.contains('/') && !s.is_empty() {
            out.insert(s.to_ascii_lowercase());
        }
    });
    let mut out: Vec<String> = out.into_iter().collect();
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// Masking
// ---------------------------------------------------------------------------

/// Values shorter than this are not masked (too many accidental matches).
pub const MIN_MASK_LEN: usize = 8;
const BASE: u64 = 1_099_511_628_211;

fn roll_hash(b: &[u8]) -> u64 {
    b.iter()
        .fold(0u64, |h, &c| h.wrapping_mul(BASE).wrapping_add(u64::from(c) + 1))
}

fn hmac(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut m = <Hmac<Sha256> as Mac>::new_from_slice(key).expect("HMAC takes any key length");
    m.update(data);
    m.finalize().into_bytes().into()
}

/// What the masker knows about one encoded form of one secret.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verifier {
    pub name: String,
    pub len: usize,
    pub rolling: u64,
    /// HMAC-SHA256(mask key, value), hex.
    pub mac: String,
}

/// The encoded forms a secret might appear in.
pub fn variants(value: &str) -> Vec<String> {
    if value.len() < MIN_MASK_LEN {
        return Vec::new();
    }
    let b = value.as_bytes();
    let mut v = vec![
        value.to_owned(),
        base64::engine::general_purpose::STANDARD.encode(b),
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b),
        url_encode(value),
    ];
    v.sort();
    v.dedup();
    v.retain(|x| x.len() >= MIN_MASK_LEN);
    v
}

fn url_encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Builds verifiers for a secret (done by whoever holds the value).
pub fn verifiers(name: &str, value: &str, mask_key: &[u8]) -> Vec<Verifier> {
    variants(value)
        .into_iter()
        .map(|v| Verifier {
            name: name.to_owned(),
            len: v.len(),
            rolling: roll_hash(v.as_bytes()),
            mac: hex(&hmac(mask_key, v.as_bytes())),
        })
        .collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Exact-match masker built from verifiers.
#[derive(Debug, Clone, Default)]
pub struct MaskSet {
    key: Vec<u8>,
    /// length → rolling hash → candidates (mac, name)
    by_len: HashMap<usize, HashMap<u64, Vec<(String, String)>>>,
    pow: HashMap<usize, u64>,
}

impl MaskSet {
    pub fn new(mask_key: Vec<u8>, verifiers: impl IntoIterator<Item = Verifier>) -> Self {
        let mut by_len: HashMap<usize, HashMap<u64, Vec<(String, String)>>> = HashMap::new();
        for v in verifiers {
            by_len
                .entry(v.len)
                .or_default()
                .entry(v.rolling)
                .or_default()
                .push((v.mac, v.name));
        }
        let pow = by_len
            .keys()
            .map(|&l| (l, (1..l).fold(1u64, |p, _| p.wrapping_mul(BASE))))
            .collect();
        MaskSet {
            key: mask_key,
            by_len,
            pow,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.by_len.is_empty()
    }

    /// Finds secret occurrences: `(start, end, name)`, non-overlapping, in order.
    pub fn find(&self, text: &str) -> Vec<(usize, usize, String)> {
        let t = text.as_bytes();
        let mut hits = Vec::new();
        for (&len, table) in &self.by_len {
            if t.len() < len {
                continue;
            }
            let pow = self.pow[&len];
            let mut h = roll_hash(&t[..len]);
            for start in 0..=t.len() - len {
                if start > 0 {
                    let out = u64::from(t[start - 1]) + 1;
                    h = h
                        .wrapping_sub(out.wrapping_mul(pow))
                        .wrapping_mul(BASE)
                        .wrapping_add(u64::from(t[start + len - 1]) + 1);
                }
                if let Some(cands) = table.get(&h) {
                    let window = &t[start..start + len];
                    let mac = hex(&hmac(&self.key, window));
                    if let Some((_, name)) = cands.iter().find(|(m, _)| *m == mac) {
                        hits.push((start, start + len, name.clone()));
                    }
                }
            }
        }
        hits.sort_by_key(|(s, e, _)| (*s, std::cmp::Reverse(*e)));
        let mut out: Vec<(usize, usize, String)> = Vec::new();
        for h in hits {
            if out.last().is_none_or(|l| h.0 >= l.1) {
                out.push(h);
            }
        }
        out
    }

    /// Replaces occurrences with handles. Returns the names found.
    pub fn mask_str(&self, s: &mut String) -> Vec<String> {
        if self.is_empty() {
            return Vec::new();
        }
        let hits = self.find(s);
        if hits.is_empty() {
            return Vec::new();
        }
        let mut out = String::with_capacity(s.len());
        let mut last = 0;
        let mut names = Vec::new();
        for (a, b, name) in hits {
            // Hits are byte ranges of exact values, so they sit on char boundaries
            // unless a value starts mid-character, which cannot happen for UTF-8 input.
            if !s.is_char_boundary(a) || !s.is_char_boundary(b) {
                continue;
            }
            out.push_str(&s[last..a]);
            out.push_str(&handle(&name));
            last = b;
            names.push(name);
        }
        out.push_str(&s[last..]);
        *s = out;
        names.sort();
        names.dedup();
        names
    }

    /// Masks every string in a JSON value. Returns the names found.
    pub fn mask_json(&self, v: &mut Value) -> Vec<String> {
        let mut names = Vec::new();
        if self.is_empty() {
            return names;
        }
        walk_mut(v, String::new(), &mut |_, s| names.extend(self.mask_str(s)));
        names.sort();
        names.dedup();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn handles_are_found_and_substituted() {
        let mut args =
            json!({"headers": {"Authorization": "Bearer {{vault:stripe_test}}"}, "body": "x {{ vault:other }}"});
        let found = find_handles(&args);
        assert_eq!(found.len(), 2);
        assert!(found.contains(&HandleUse {
            name: "stripe_test".into(),
            path: "headers.Authorization".into()
        }));
        substitute(&mut args, |n, _| (n == "stripe_test").then(|| "sk_test_123".to_owned()));
        assert_eq!(args["headers"]["Authorization"], "Bearer sk_test_123");
        assert_eq!(args["body"], "x {{ vault:other }}");
        assert!(!valid_name("Bad Name") && valid_name("stripe_test.v2"));
    }

    #[test]
    fn destinations_from_urls_and_host_fields() {
        let d = destinations(
            &json!({"url": "https://api.stripe.com/v1/invoices?x=1", "alt": "see http://evil.example:8080/x", "host": "db.internal"}),
        );
        assert_eq!(d, ["api.stripe.com", "db.internal", "evil.example"]);
    }

    #[test]
    fn masks_raw_base64_and_urlencoded_forms_without_plaintext() {
        let key = b"org-mask-key".to_vec();
        let secret = "sk_test_51Habc/DEF+ghi=123";
        let ms = MaskSet::new(key, verifiers("stripe_test", secret, b"org-mask-key"));
        let b64 = base64::engine::general_purpose::STANDARD.encode(secret);
        let mut text = format!(
            "key={secret} b64={b64} url={} other=sk_test_nothere",
            url_encode(secret)
        );
        let names = ms.mask_str(&mut text);
        assert_eq!(names, ["stripe_test"]);
        assert!(!text.contains(secret) && !text.contains(&b64));
        assert_eq!(text.matches("{{vault:stripe_test}}").count(), 3);
        // A different key's verifiers don't match.
        let other = MaskSet::new(b"k2".to_vec(), verifiers("x", secret, b"k1"));
        assert!(other.find(secret).is_empty());
        // Short values are never masked.
        assert!(verifiers("pin", "1234", b"k").is_empty());
    }
}
