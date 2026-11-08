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
            // verb, up to four filler words, then a noun that names instructions.
            // Weak nouns (rules, directions, text…) need a "previous"-type word.
            concat!(
                r"(?i)\b(?:ignore|ignoring|disregard|forget|override|discard|bypass|abandon|(?:do\s+not|don'?t|never)\s+(?:follow|obey))",
                r"(?:\s+(?:all|any|every|each|of|the|your|my|these|those|given|previous|previously|prior|above|earlier|preceding|original|initial|old|existing|system|safety|about|you\s+(?:have|got|were|received)))",
                r"{0,4}\s+(?:instru\w*|intru\w*|istru\w*|prompts?|directives?|guidelines|restrictions?|constraints|programming)\b",
                r"|(?i)\b(?:ignore|disregard|forget|override|discard|(?:do\s+not|don'?t)\s+follow)\s+(?:all\s+|any\s+|the\s+|your\s+)*(?:previous|prior|above|earlier|preceding|original|initial|given)\s+(?:\w+\s+)?(?:rules|directions|commands|orders|context|tasks|assignments|requests|text|information|answers|messages|expressions|techniques)\b",
                r"|(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+|the\s+)?(?:instructions|directions|rules|text|orders)\s+above\b",
                r"|(?i)\b(?:delete|remove|erase|clear)\s+(?:all\s+|your\s+|the\s+)*(?:previous\s+|prior\s+)?instructions\b",
                r"|(?i)\b(?:ignore|disregard|forget)\s+(?:about\s+)?(?:the\s+)?(?:above|everything(?:\s+(?:above|before|else|you\s+know))?|all\s+(?:of\s+)?(?:the\s+)?above)\b",
                // German, French, Spanish, Italian, Portuguese.
                r"|(?i)\b(?:ignorier\w*|vergiss|vergessen\s+sie)\s+(?:\w+\s+){0,3}(?:anweisungen|instruktionen|befehle|aufgaben|auftr[äa]ge|angaben|vorgaben|alles(?:\s+(?:davor|vorher|oben|bisherige))?)\b",
                r"|(?i)\b(?:ignore[zr]?|oublie[zr]?)\s+(?:\w+\s+){0,3}(?:instructions|consignes|directives)\b",
                r"|(?i)\b(?:ignora|ignore|olvid[ae]r?|dimentica|esque[cç]a)\s+(?:\w+\s+){0,3}(?:instrucciones|indicaciones|istruzioni|instru[cç][oõ]es)\b",
            ),
        ),
        rule(
            "injection.prompt_leak",
            Action::Annotate,
            "asks the agent to reveal its own instructions",
            concat!(
                r"(?i)\b(?:reveal|repeat|print|output|quote|show|list|display|leak|dump|recite|spell\s+out|tell\s+me)\s+(?:me\s+)?(?:all\s+|every\s+)?(?:of\s+)?",
                r"(?:your|the\s+(?:system|secret|hidden|initial|original))\s+(?:\w+\s+){0,2}(?:instructions?|prompts?|guidelines|directives|system\s+prompt)\b",
                r"|(?i)\bwhat\s+(?:are|were|is)\s+(?:all\s+)?your\s+(?:\w+\s+){0,2}(?:instructions?|system\s+prompt)\b",
                r"|(?i)\breveal\s+(?:\w+\s+)?(?:secret|hidden|system)\s+(?:instructions|prompt)\b",
                r"|(?i)\b(?:repeat|output|print|list|recite|tell\s+me)\s+(?:all\s+)?(?:of\s+)?(?:the\s+)?(?:previous|prior|initial|original|given|secret|hidden)\s+instructions\b",
                r"|(?i)\binstructions?\s+(?:(?:you|u)\s+(?:have\s+been\s+given|were\s+given|got|received|were\s+u\s+give)|(?:given|provided)\s+to\s+you)\b",
            ),
        ),
        rule(
            "injection.new_task",
            Action::Annotate,
            "announces replacement instructions for the agent",
            concat!(
                r"(?i)\b(?:focus\s+on|concentrate\s+on|start\s+(?:over\s+)?with|perform)\s+(?:your|a|the)\s+(?:new|different)\s+task\b",
                r"|(?i)\bkonzentriere\s+dich\s+(?:jetzt\s+|nun\s+)?auf\s+(?:deine|die)\s+neue\s+aufgabe\b",
                r"|(?i)\bnew\s+(?:instructions|tasks)\s+follow\b",
                r"|(?i)\bi\s+(?:give|am\s+giving|have)\s+(?:you\s+)?(?:new|different|updated)\s+(?:instructions|guidelines|orders)\b",
                r"|(?i)\binstructions\s+that\s+override\b",
                r"|(?i)\b(?:nun|jetzt)\s+folgen\s+(?:neue|weitere)\s+(?:aufgaben|anweisungen)\b",
            ),
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
            concat!(
                r"(?i)\byou\s+are\s+now\s+(?:a|an|in|the|my)\b|\bfrom\s+now\s+on,?\s+you\s+(?:will|must|are)\b",
                // "You are now Frodo", "Now you are Ted": a capitalised name.
                r"|(?i:\byou\s+are\s+now|\bnow\s+you\s+are)\s+[A-Z][a-z]",
                r"|(?i)\bi\s+want\s+you\s+to\s+act\s+as\b|\bpretend\s+(?:to\s+be|you\s+(?:are|can|have))\b|\bact\s+as\s+if\s+you\b",
                r"|(?i)\b(?:jetzt|nun)\s+bist\s+du\b|\bstell\s+dir\s+vor,?\s+du\s+bist\b|\bich\s+m[öo]chte,?\s+dass\s+(?:sie|du)\s+als\s+\S+\s+fungier",
            ),
        ),
        rule(
            "injection.tool_coercion",
            Action::Annotate,
            "instructs the agent to call a specific tool",
            r#"(?i)\b(?:call|invoke|use|run)\s+the\s+[`"']?[a-z0-9_.-]+[`"']?\s+tool\b"#,
        ),
        rule(
            "injection.credential_exfil",
            Action::Annotate,
            "asks the agent to send credentials somewhere",
            r#"(?i)\b(?:send|post|upload|forward|email|paste|share)\s+(?:the|all|your|any|all\s+(?:the|your))\s+(?:secrets?|credentials|api\s+keys?|tokens?|passwords?|\.env|ssh\s+keys?|private\s+keys?)"#,
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
