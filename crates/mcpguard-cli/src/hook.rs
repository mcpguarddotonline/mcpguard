//! `mcpguard hook prompt`: a prompt hook for clients that support one
//! (Claude Code `UserPromptSubmit`). It reads the hook's JSON on stdin and
//! blocks the prompt (exit code 2, reason on stderr) when it contains a
//! secret: a known key format, or a value from the local vault. Everything
//! runs locally; nothing is sent anywhere.

use std::io::Read;

use anyhow::Result;
use mcpguard_engine::vault::MaskSet;
use mcpguard_engine::{Direction, Kind, inspect};
use serde_json::Value;

use crate::vault::Store;

/// Returns why the prompt must be blocked, if it must.
pub fn check_prompt(prompt: &str, store: &dyn Store) -> Option<String> {
    let v = inspect(prompt, Direction::ToServer);
    let kinds: Vec<&str> = v
        .findings
        .iter()
        .filter(|f| f.kind == Kind::Secret)
        .filter_map(|f| f.label)
        .collect();
    if !kinds.is_empty() {
        return Some(format!(
            "MCP-Guard blocked this prompt: it contains a secret ({}). Store it with `mcpguard vault add <name>` and refer to it as {{{{vault:<name>}}}} instead.",
            kinds.join(", ")
        ));
    }
    let key = b"prompt-hook".to_vec();
    let mut verifiers = Vec::new();
    for name in store.names().unwrap_or_default() {
        if let Ok(value) = store.get(&name) {
            verifiers.extend(mcpguard_engine::vault::verifiers(&name, &value, &key));
        }
    }
    let found: Vec<String> = MaskSet::new(key, verifiers)
        .find(prompt)
        .into_iter()
        .map(|h| h.2)
        .collect();
    (!found.is_empty()).then(|| {
        format!("MCP-Guard blocked this prompt: it contains the value of vault secret '{}'. Use {{{{vault:{}}}}} instead, and consider rotating it.", found[0], found[0])
    })
}

pub fn prompt(store: &dyn Store) -> Result<i32> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let v: Value = serde_json::from_str(&input).unwrap_or(Value::Null);
    let prompt = v.get("prompt").and_then(Value::as_str).unwrap_or(&input);
    match check_prompt(prompt, store) {
        Some(reason) => {
            eprintln!("{reason}");
            Ok(2)
        }
        None => Ok(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::Memory;

    #[test]
    fn blocks_known_formats_and_vault_values() {
        let mut m = Memory::default();
        m.set("db_password", "correct-horse-battery-staple").expect("set");
        let gh = ["ghp_", &"A1b2C3d4E5f6".repeat(3)].concat();
        assert!(check_prompt(&format!("use token {gh} please"), &m).is_some());
        assert!(
            check_prompt("the password is correct-horse-battery-staple", &m)
                .unwrap_or_default()
                .contains("db_password")
        );
        assert!(check_prompt("summarize this file", &m).is_none());
    }
}
