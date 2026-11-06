//! Layer L1: prompt-injection signatures for content the model will read.
//!
//! `Annotate` marks content as untrusted data; `Block` is reserved for patterns
//! with almost no legitimate use inside tool output. The ML classifier (L2)
//! will plug in behind the same [`Finding`] type.

use std::sync::LazyLock;

use regex::Regex;

use crate::verdict::{Action, Finding, Kind};

struct InjectionRule {
    id: &'static str,
    action: Action,
    note: &'static str,
    re: Regex,
}

fn rule(id: &'static str, action: Action, note: &'static str, pattern: &str) -> InjectionRule {
    InjectionRule {
        id,
        action,
        note,
        re: Regex::new(pattern).expect("built-in injection rule must compile"),
    }
}

static RULES: LazyLock<Vec<InjectionRule>> = LazyLock::new(|| {
    vec![
        rule(
            "injection.ignore_instructions",
            Action::Annotate,
            "tries to override the agent's instructions",
            r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|preceding|your)\s+(?:instructions|prompts?|rules|directions|guidelines)",
        ),
        rule(
            "injection.role_marker",
            Action::Annotate,
            "fake chat-role or system markers",
            r"(?im)<\|im_start\|>|<\|im_end\|>|</?system>|\[/?INST\]|<<SYS>>|^\s*(?:system|assistant)\s*:",
        ),
        rule(
            "injection.new_identity",
            Action::Annotate,
            "tries to reassign the agent's role",
            r"(?i)\byou\s+are\s+now\s+(?:a|an|in|the|my)\b|\bfrom\s+now\s+on,?\s+you\s+(?:will|must|are)\b",
        ),
        rule(
            "injection.tool_coercion",
            Action::Annotate,
            "instructs the agent to call tools or send credentials",
            r#"(?i)\b(?:call|invoke|use|run)\s+the\s+[`"']?[a-z0-9_.-]+[`"']?\s+tool\b|\b(?:send|post|upload|forward|email)\s+(?:the|all|your|any)\s+(?:secrets?|credentials|api\s+keys?|tokens?|passwords?|\.env|ssh\s+keys?)"#,
        ),
        rule(
            "injection.markdown_exfil",
            Action::Block,
            "markdown image with data in the query string (zero-click exfiltration)",
            r"!\[[^\]]*\]\(\s*https?://[^)\s]+\?[^)\s]*=[^)\s]{8,}\)",
        ),
    ]
});

pub(crate) fn find_injection(text: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    for r in RULES.iter() {
        for m in r.re.find_iter(text) {
            out.push(Finding {
                rule_id: r.id,
                kind: Kind::Injection,
                action: r.action,
                start: m.start(),
                end: m.end(),
                label: None,
                note: r.note,
            });
        }
    }
    out
}
