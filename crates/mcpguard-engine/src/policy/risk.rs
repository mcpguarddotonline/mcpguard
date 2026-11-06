//! How risky is a tool? Combines a curated catalog, the server's own MCP
//! annotations (hints only: a server can lie, so they can raise risk but
//! never lower what the tool's name says), and verb heuristics.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Risk {
    Read,
    Unknown,
    Write,
    Destructive,
}

const DESTRUCTIVE: &[&str] = &[
    "delete",
    "remove",
    "rm",
    "drop",
    "destroy",
    "truncate",
    "purge",
    "wipe",
    "erase",
    "revoke",
    "terminate",
    "kill",
    "reset",
    "force",
    "overwrite",
    "uninstall",
    "deactivate",
    "cancel",
    "refund",
    "archive",
    "close",
    "ban",
];
const WRITE: &[&str] = &[
    "create", "update", "write", "edit", "set", "add", "insert", "put", "post", "send", "push", "merge", "move",
    "rename", "upload", "publish", "deploy", "execute", "exec", "run", "invoke", "apply", "commit", "comment", "reply",
    "assign", "approve", "transfer", "pay", "charge", "invite", "schedule", "patch", "replace", "fork",
];
const READ: &[&str] = &[
    "get", "list", "search", "read", "fetch", "find", "query", "describe", "view", "show", "lookup", "retrieve",
    "count", "check", "inspect", "download", "browse", "status", "whoami", "me", "info", "explain", "diff", "log",
];

/// Curated entries for well-known servers' tools (by tool name).
const CATALOG: &[(&str, Risk)] = &[
    ("merge_pull_request", Risk::Write),
    ("push_files", Risk::Write),
    ("create_or_update_file", Risk::Write),
    ("delete_file", Risk::Destructive),
    ("delete_repository", Risk::Destructive),
    ("execute_sql", Risk::Unknown), // the SQL decides
    ("query", Risk::Unknown),
    ("run_command", Risk::Unknown), // the command decides
    ("execute_command", Risk::Unknown),
    ("shell", Risk::Unknown),
    ("write_file", Risk::Write),
    ("edit_file", Risk::Write),
    ("move_file", Risk::Write),
    ("create_refund", Risk::Destructive),
    ("cancel_subscription", Risk::Destructive),
    ("create_payment_intent", Risk::Write),
    ("slack_post_message", Risk::Write),
    ("send_email", Risk::Write),
];

/// Splits `deleteRepo`, `delete_repo`, `delete-repo`, `repo.delete` into words.
pub fn words(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for c in name.chars() {
        if !c.is_alphanumeric() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            prev_lower = false;
            continue;
        }
        if c.is_uppercase() && prev_lower && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        prev_lower = c.is_lowercase() || c.is_ascii_digit();
        cur.extend(c.to_lowercase());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn by_verbs(tool: &str) -> Risk {
    let w = words(tool);
    let has = |list: &[&str]| w.iter().any(|x| list.contains(&x.as_str()));
    if has(DESTRUCTIVE) {
        Risk::Destructive
    } else if has(WRITE) {
        Risk::Write
    } else if has(READ) {
        Risk::Read
    } else {
        Risk::Unknown
    }
}

fn by_annotations(def: &Value) -> Option<Risk> {
    let a = def.get("annotations")?;
    let flag = |k: &str| a.get(k).and_then(Value::as_bool);
    match (flag("destructiveHint"), flag("readOnlyHint")) {
        (Some(true), _) => Some(Risk::Destructive),
        (_, Some(true)) => Some(Risk::Read),
        (Some(false), Some(false)) => Some(Risk::Write),
        _ => None,
    }
}

/// Classifies a tool. Returns the risk and where it came from.
pub fn classify(tool: &str, def: Option<&Value>) -> (Risk, &'static str) {
    if let Some((_, r)) = CATALOG.iter().find(|(name, _)| *name == tool) {
        return (*r, "catalog");
    }
    let verbs = by_verbs(tool);
    match def.and_then(by_annotations) {
        // Annotations may raise risk, never lower it below the verbs'.
        Some(a) if a > verbs => (a, "annotations"),
        Some(Risk::Read) if verbs == Risk::Unknown => (Risk::Read, "annotations"),
        _ => (verbs, "name"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classification() {
        assert_eq!(classify("delete_repository", None).0, Risk::Destructive);
        assert_eq!(classify("deleteRepo", None).0, Risk::Destructive);
        assert_eq!(classify("list_issues", None).0, Risk::Read);
        assert_eq!(classify("create_issue", None).0, Risk::Write);
        assert_eq!(classify("frobnicate", None).0, Risk::Unknown);
        // A server claiming read-only can't hide a delete.
        let lie = json!({"annotations": {"readOnlyHint": true}});
        assert_eq!(classify("delete_branch", Some(&lie)).0, Risk::Destructive);
        // But it can raise risk.
        let honest = json!({"annotations": {"destructiveHint": true}});
        assert_eq!(classify("sync", Some(&honest)).0, Risk::Destructive);
        assert_eq!(words("getUserID-v2.list"), ["get", "user", "id", "v2", "list"]);
    }
}
