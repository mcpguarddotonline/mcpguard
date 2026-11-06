//! Layer L0: normalization. Strips characters that render as nothing but can
//! carry instructions a human reviewer never sees.
//!
//! Legitimate text uses some of these characters too, so the rules are
//! deliberately narrow:
//! - **Unicode tag characters** (U+E0000–E007F) have no use in tool output;
//!   any run is removed and blocked (this is the "ASCII smuggling" channel).
//! - **Zero-width characters** (U+200B–U+200D, U+2060, U+FEFF) are removed and
//!   blocked when two or more appear in a row (binary-encoded payloads). A
//!   single zero-width joiner / non-joiner is kept silently: emoji sequences
//!   and Indic/Persian scripts need it. A leading byte-order mark is dropped
//!   silently.
//! - **Bidirectional controls** (U+202A–U+202E, U+2066–U+2069) are kept, since
//!   right-to-left text needs them, but flagged so the content is annotated
//!   ("Trojan Source" reordering).

use std::borrow::Cow;

use crate::verdict::{Action, Finding, Kind};

/// Tag characters and zero-width characters.
fn is_hidden(c: char) -> bool {
    matches!(c as u32, 0xE0000..=0xE007F | 0x200B..=0x200D | 0x2060 | 0xFEFF)
}

fn is_bidi(c: char) -> bool {
    matches!(c as u32, 0x202A..=0x202E | 0x2066..=0x2069)
}

/// Returns the text with hidden runs removed, plus one finding per run
/// (offsets into `text`). Borrows when nothing needed removing.
pub(crate) fn normalize(text: &str) -> (Cow<'_, str>, Vec<Finding>) {
    if !text.chars().any(|c| is_hidden(c) || is_bidi(c)) {
        return (Cow::Borrowed(text), Vec::new());
    }
    let mut out = String::with_capacity(text.len());
    let mut findings = Vec::new();
    let mut changed = false;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if is_bidi(c) {
            findings.push(finding(
                "injection.bidi_control",
                Action::Annotate,
                i,
                i + c.len_utf8(),
                "bidirectional control characters can reorder how text is displayed",
            ));
            out.push(c);
            continue;
        }
        if !is_hidden(c) {
            out.push(c);
            continue;
        }
        // Collect the whole run of hidden characters.
        let mut end = i + c.len_utf8();
        let mut len = 1;
        while let Some(&(j, d)) = chars.peek() {
            if !is_hidden(d) {
                break;
            }
            len += 1;
            end = j + d.len_utf8();
            chars.next();
        }
        let single_joiner = len == 1 && matches!(c as u32, 0x200C | 0x200D);
        let leading_bom = len == 1 && i == 0 && c == '\u{FEFF}';
        if single_joiner {
            out.push(c);
        } else if leading_bom {
            changed = true;
        } else {
            changed = true;
            findings.push(finding(
                "injection.hidden_unicode",
                Action::Block,
                i,
                end,
                "invisible Unicode characters (tags / zero-width runs) can hide instructions",
            ));
        }
    }
    if changed {
        (Cow::Owned(out), findings)
    } else {
        (Cow::Borrowed(text), findings)
    }
}

fn finding(rule_id: &'static str, action: Action, start: usize, end: usize, note: &'static str) -> Finding {
    Finding {
        rule_id,
        kind: Kind::Injection,
        action,
        start,
        end,
        label: None,
        note,
    }
}
