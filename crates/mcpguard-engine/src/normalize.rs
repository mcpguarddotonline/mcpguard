//! Layer L0: normalization. Strips characters that render as nothing but can
//! carry instructions a human reviewer never sees.

use std::borrow::Cow;

use crate::verdict::{Action, Finding, Kind};

/// Unicode tag characters (U+E0000–E007F), zero-width characters, the BOM and
/// bidirectional controls.
pub(crate) fn is_invisible(c: char) -> bool {
    matches!(
        c as u32,
        0xE0000..=0xE007F | 0x200B..=0x200D | 0x2060 | 0xFEFF | 0x202A..=0x202E | 0x2066..=0x2069
    )
}

/// Returns the text without invisible characters, plus one finding per hidden
/// run (offsets into `text`). Borrows when nothing needed removing.
pub(crate) fn normalize(text: &str) -> (Cow<'_, str>, Vec<Finding>) {
    if !text.chars().any(is_invisible) {
        return (Cow::Borrowed(text), Vec::new());
    }
    let mut out = String::with_capacity(text.len());
    let mut findings = Vec::new();
    let mut run_start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if is_invisible(c) {
            run_start.get_or_insert(i);
            continue;
        }
        if let Some(s) = run_start.take() {
            findings.push(hidden(s, i));
        }
        out.push(c);
    }
    if let Some(s) = run_start {
        findings.push(hidden(s, text.len()));
    }
    (Cow::Owned(out), findings)
}

fn hidden(start: usize, end: usize) -> Finding {
    Finding {
        rule_id: "injection.hidden_unicode",
        kind: Kind::Injection,
        action: Action::Block,
        start,
        end,
        label: None,
        note: "invisible Unicode characters (tags / zero-width / bidi) can hide instructions",
    }
}
