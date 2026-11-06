//! `mcpguard install` / `uninstall`: put every stdio MCP server in your
//! clients behind `mcpguard wrap`, move plaintext secrets into the local
//! vault, add the prompt hook. Every file is backed up first; `uninstall`
//! restores the backups.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::clients::{self, Format};
use crate::scan::plaintext_secret;
use crate::vault::{Store, config_dir};

#[derive(Serialize, Deserialize, Default)]
struct Ledger {
    /// (config file, backup file)
    backups: Vec<(PathBuf, PathBuf)>,
}

fn ledger_path() -> PathBuf {
    config_dir().join("installs.json")
}

fn load_ledger() -> Ledger {
    std::fs::read_to_string(ledger_path())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

fn save_ledger(l: &Ledger) -> Result<()> {
    std::fs::create_dir_all(config_dir())?;
    std::fs::write(ledger_path(), serde_json::to_vec_pretty(l)?).context("writing install ledger")
}

fn secret_name(server: &str, key: &str) -> String {
    let raw = format!("{server}_{key}").to_ascii_lowercase();
    let n: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    n.trim_matches('_').chars().take(64).collect()
}

/// Rewrites one server entry in place. Returns a description of the change.
pub fn wrap_entry(server: &str, def: &mut Value, exe: &str, store: &mut dyn Store) -> Result<Option<String>> {
    let Some(cmd) = def.get("command").and_then(Value::as_str).map(str::to_owned) else {
        return Ok(None); // remote (url) server: nothing to wrap locally
    };
    if Path::new(&cmd).file_name().and_then(|x| x.to_str()) == Some("mcpguard") || cmd == exe {
        return Ok(None);
    }
    let args: Vec<String> = def
        .get("args")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect())
        .unwrap_or_default();
    let mut new_args = vec!["wrap".to_owned(), "--name".to_owned(), server.to_owned()];
    let mut moved = Vec::new();
    if let Some(env) = def.get_mut("env").and_then(Value::as_object_mut) {
        let keys: Vec<String> = env.keys().cloned().collect();
        for k in keys {
            let Some(v) = env.get(&k).and_then(Value::as_str).map(str::to_owned) else {
                continue;
            };
            if plaintext_secret(&k, &v) {
                let name = secret_name(server, &k);
                store.set(&name, &v)?;
                env.remove(&k);
                new_args.push("--env".into());
                new_args.push(format!("{k}=vault:{name}"));
                moved.push(name);
            }
        }
    }
    new_args.push("--".into());
    new_args.push(cmd.clone());
    new_args.extend(args);
    def["command"] = json!(exe);
    def["args"] = json!(new_args);
    let mut d = format!("{server}: now runs behind mcpguard wrap");
    if !moved.is_empty() {
        d.push_str(&format!("; moved {} into the vault", moved.join(", ")));
    }
    Ok(Some(d))
}

fn add_claude_code_hook(exe: &str, dry: bool) -> Result<Option<String>> {
    let path = dirs::home_dir().unwrap_or_default().join(".claude/settings.json");
    let mut v: Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| json!({}));
    let cmd = format!("{exe} hook prompt");
    if v.to_string().contains("hook prompt") {
        return Ok(None);
    }
    let hooks = v
        .as_object_mut()
        .context("settings.json is not an object")?
        .entry("hooks")
        .or_insert_with(|| json!({}));
    let list = hooks
        .as_object_mut()
        .context("hooks is not an object")?
        .entry("UserPromptSubmit")
        .or_insert_with(|| json!([]));
    list.as_array_mut()
        .context("UserPromptSubmit is not a list")?
        .push(json!({"hooks": [{"type": "command", "command": cmd}]}));
    if !dry {
        backup(&path)?;
        std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
        std::fs::write(&path, serde_json::to_vec_pretty(&v)?)?;
    }
    Ok(Some(format!("Claude Code: prompt hook added to {}", path.display())))
}

fn backup(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let b = path.with_extension(format!("mcpguard-backup-{stamp}"));
    std::fs::copy(path, &b).with_context(|| format!("backing up {}", path.display()))?;
    let mut l = load_ledger();
    l.backups.push((path.to_owned(), b));
    save_ledger(&l)
}

pub fn install(project: Option<&Path>, dry: bool, store: &mut dyn Store) -> Result<i32> {
    let exe = std::env::current_exe()?.display().to_string();
    let mut changes = Vec::new();
    for f in clients::discover(project) {
        if f.format == Format::CodexToml {
            changes.push(format!(
                "{}: edit by hand for now (prefix each command with `mcpguard wrap --`)",
                f.path.display()
            ));
            continue;
        }
        let text = std::fs::read_to_string(&f.path)?;
        let mut v: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                changes.push(format!("{}: skipped, not valid JSON ({e})", f.path.display()));
                continue;
            }
        };
        let key = if f.format == Format::Servers {
            "servers"
        } else {
            "mcpServers"
        };
        let mut file_changes = Vec::new();
        let mut process = |m: &mut serde_json::Map<String, Value>, store: &mut dyn Store| -> Result<()> {
            for (name, def) in m.iter_mut() {
                if dry {
                    let mut probe = def.clone();
                    let mut mem = crate::vault::Memory::default();
                    if let Some(c) = wrap_entry(name, &mut probe, &exe, &mut mem)? {
                        file_changes.push(c);
                    }
                } else if let Some(c) = wrap_entry(name, def, &exe, store)? {
                    file_changes.push(c);
                }
            }
            Ok(())
        };
        if let Some(m) = v.get_mut(key).and_then(Value::as_object_mut) {
            process(m, store)?;
        }
        if let Some(projects) = v.get_mut("projects").and_then(Value::as_object_mut) {
            for pv in projects.values_mut() {
                if let Some(m) = pv.get_mut("mcpServers").and_then(Value::as_object_mut) {
                    process(m, store)?;
                }
            }
        }
        if file_changes.is_empty() {
            continue;
        }
        if !dry {
            backup(&f.path)?;
            std::fs::write(&f.path, serde_json::to_vec_pretty(&v)?)?;
        }
        for c in file_changes {
            changes.push(format!("{} · {c}", f.client));
        }
    }
    if let Some(h) = add_claude_code_hook(&exe, dry)? {
        changes.push(h);
    }
    if changes.is_empty() {
        println!("Nothing to change: no unprotected stdio servers found.");
    } else {
        println!("{}", if dry { "Would change:" } else { "Changed:" });
        for c in &changes {
            println!("  - {c}");
        }
        if !dry {
            println!("\nRestart your AI clients. Undo everything with `mcpguard uninstall`.");
        }
    }
    Ok(0)
}

pub fn uninstall() -> Result<i32> {
    let mut l = load_ledger();
    if l.backups.is_empty() {
        println!("Nothing to undo.");
        return Ok(0);
    }
    // Restore the oldest backup of each file (the state before any install).
    let mut seen = std::collections::HashSet::new();
    for (file, b) in &l.backups {
        if seen.insert(file.clone()) {
            std::fs::copy(b, file).with_context(|| format!("restoring {}", file.display()))?;
            println!("restored {}", file.display());
        }
    }
    l.backups.clear();
    save_ledger(&l)?;
    println!("Secrets stay in your local vault (`mcpguard vault list`); remove them with `mcpguard vault rm`.");
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::Memory;

    #[test]
    fn wraps_and_moves_plaintext_secrets() {
        let tok = ["ghp_", &"A1b2C3d4E5f6".repeat(3)].concat();
        let mut def = json!({"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"], "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": tok, "LOG_LEVEL": "debug"}});
        let mut m = Memory::default();
        let change = wrap_entry("github", &mut def, "/usr/local/bin/mcpguard", &mut m)
            .expect("wrap")
            .expect("changed");
        assert!(change.contains("github_github_personal_access_token"));
        assert_eq!(def["command"], "/usr/local/bin/mcpguard");
        assert_eq!(
            def["args"],
            json!([
                "wrap",
                "--name",
                "github",
                "--env",
                "GITHUB_PERSONAL_ACCESS_TOKEN=vault:github_github_personal_access_token",
                "--",
                "npx",
                "-y",
                "@modelcontextprotocol/server-github"
            ])
        );
        assert_eq!(def["env"], json!({"LOG_LEVEL": "debug"}));
        assert_eq!(m.get("github_github_personal_access_token").expect("stored"), tok);
        // Running install twice changes nothing.
        assert!(
            wrap_entry("github", &mut def, "/usr/local/bin/mcpguard", &mut m)
                .expect("wrap")
                .is_none()
        );
        // Remote servers are left alone.
        let mut remote = json!({"url": "https://mcp.linear.app/mcp"});
        assert!(
            wrap_entry("linear", &mut remote, "/x/mcpguard", &mut m)
                .expect("wrap")
                .is_none()
        );
    }
}
