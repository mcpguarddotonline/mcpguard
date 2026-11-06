//! `mcpguard scan`: audit the MCP configs on this machine.

use std::io::Write;
use std::path::Path;

use anyhow::Result;
use mcpguard_engine::{Direction, Kind, inspect};
use serde::Serialize;

use crate::clients::{self, Server};

#[derive(Debug, Serialize, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Info,
    Warn,
    High,
}

#[derive(Debug, Serialize, Clone)]
pub struct Issue {
    pub level: Level,
    pub client: String,
    pub file: String,
    pub server: String,
    pub code: &'static str,
    pub message: String,
}

const SECRETISH: &[&str] = &[
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "AUTH",
    "CREDENTIAL",
    "PAT",
    "COOKIE",
];

/// Is this value a plaintext secret? (Known key formats, or a secret-named
/// variable with a non-trivial value that isn't already a vault reference.)
pub fn plaintext_secret(key: &str, value: &str) -> bool {
    if value.contains("vault:") || value.starts_with("${") || value.len() < 8 {
        return false;
    }
    let named = SECRETISH.iter().any(|w| key.to_ascii_uppercase().contains(w));
    let pattern = inspect(value, Direction::ToServer)
        .findings
        .iter()
        .any(|f| f.kind == Kind::Secret);
    pattern || (named && !value.contains(' ') && !value.starts_with('/'))
}

pub fn check_server(client: &str, file: &str, s: &Server) -> Vec<Issue> {
    let mut out = Vec::new();
    let mut push = |level, code, message: String| {
        out.push(Issue {
            level,
            client: client.into(),
            file: file.into(),
            server: s.name.clone(),
            code,
            message,
        })
    };
    for (k, v) in &s.env {
        if plaintext_secret(k, v) {
            push(
                Level::High,
                "plaintext_secret",
                format!("env {k} holds a plaintext secret: move it into the vault (`mcpguard install` does it)"),
            );
        }
    }
    for (k, v) in &s.headers {
        let token = v.strip_prefix("Bearer ").unwrap_or(v);
        if plaintext_secret(k, token) {
            push(
                Level::High,
                "plaintext_secret",
                format!("header {k} holds a plaintext credential"),
            );
        }
    }
    for a in &s.args {
        if inspect(a, Direction::ToServer)
            .findings
            .iter()
            .any(|f| f.kind == Kind::Secret)
        {
            push(
                Level::High,
                "secret_in_args",
                "a command-line argument contains a secret (visible to every process on the machine)".into(),
            );
        }
    }
    if let Some(cmd) = &s.command {
        let base = Path::new(cmd).file_name().and_then(|x| x.to_str()).unwrap_or(cmd);
        if base == "mcpguard" {
            push(Level::Info, "protected", "already runs behind mcpguard wrap".into());
            return out;
        }
        if matches!(base, "npx" | "bunx" | "uvx" | "pnpx") {
            let pkg = s.args.iter().find(|a| !a.starts_with('-'));
            if let Some(p) = pkg {
                let pinned = p
                    .rsplit_once('@')
                    .is_some_and(|(name, ver)| !name.is_empty() && !ver.is_empty() && ver != "latest");
                if !pinned {
                    push(
                        Level::Warn,
                        "unpinned_package",
                        format!("{base} runs `{p}` without a pinned version: a new release runs automatically"),
                    );
                }
            }
        }
        if matches!(base, "docker" | "podman")
            && s.args
                .iter()
                .any(|a| a == "--privileged" || a.contains("/var/run/docker.sock"))
        {
            push(
                Level::High,
                "container_escape",
                "container gets privileged access to the host".into(),
            );
        }
        push(
            Level::Info,
            "unprotected",
            "not protected yet: `mcpguard install` wraps it".into(),
        );
    }
    if let Some(u) = &s.url {
        if u.starts_with("http://") && !u.contains("localhost") && !u.contains("127.0.0.1") {
            push(Level::High, "plain_http", format!("{u} is not encrypted (http://)"));
        }
    }
    out
}

pub fn run(project: Option<&Path>, json: bool) -> Result<i32> {
    let files = clients::discover(project);
    let mut issues = Vec::new();
    let mut servers = 0;
    for f in &files {
        match clients::read(f) {
            Ok(list) => {
                servers += list.len();
                for s in &list {
                    issues.extend(check_server(f.client, &f.path.display().to_string(), s));
                }
            }
            Err(e) => issues.push(Issue {
                level: Level::Warn,
                client: f.client.into(),
                file: f.path.display().to_string(),
                server: "-".into(),
                code: "unreadable",
                message: format!("{e:#}"),
            }),
        }
    }
    let high = issues.iter().filter(|i| i.level == Level::High).count();
    let mut out = std::io::stdout().lock();
    if json {
        serde_json::to_writer_pretty(
            &mut out,
            &serde_json::json!({"configs": files.iter().map(|f| f.path.display().to_string()).collect::<Vec<_>>(), "servers": servers, "issues": issues}),
        )?;
        writeln!(out)?;
    } else {
        writeln!(
            out,
            "Scanned {} config file(s), {servers} MCP server(s).\n",
            files.len()
        )?;
        if files.is_empty() {
            writeln!(
                out,
                "No MCP client configs found (Claude Desktop/Code, Cursor, VS Code, Codex, Windsurf)."
            )?;
        }
        issues.sort_by(|a, b| b.level.cmp(&a.level));
        for i in &issues {
            let tag = match i.level {
                Level::High => "HIGH",
                Level::Warn => "WARN",
                Level::Info => "info",
            };
            writeln!(out, "[{tag}] {} · {}: {}", i.client, i.server, i.message)?;
        }
        if high > 0 {
            writeln!(
                out,
                "\n{high} high-risk issue(s). Run `mcpguard install` to move secrets into the vault and put servers behind MCP-Guard."
            )?;
        } else if !issues.is_empty() {
            writeln!(out, "\nNo high-risk issues.")?;
        }
    }
    Ok(if high > 0 { 2 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server(env: &[(&str, &str)], cmd: &str, args: &[&str]) -> Server {
        Server {
            name: "s".into(),
            command: Some(cmd.into()),
            args: args.iter().map(|s| s.to_string()).collect(),
            env: env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            url: None,
            headers: vec![],
        }
    }

    #[test]
    fn finds_plaintext_tokens_and_unpinned_packages() {
        let tok = ["ghp_", &"A1b2C3d4E5f6".repeat(3)].concat();
        let s = server(
            &[("GITHUB_PERSONAL_ACCESS_TOKEN", &tok), ("LOG_LEVEL", "debug")],
            "npx",
            &["-y", "@modelcontextprotocol/server-github"],
        );
        let codes: Vec<&str> = check_server("c", "f", &s).iter().map(|i| i.code).collect();
        assert!(codes.contains(&"plaintext_secret"));
        assert!(codes.contains(&"unpinned_package"));
        let pinned = server(
            &[("GITHUB_TOKEN", "vault:github")],
            "npx",
            &["-y", "@modelcontextprotocol/server-github@2025.4.8"],
        );
        let codes: Vec<&str> = check_server("c", "f", &pinned).iter().map(|i| i.code).collect();
        assert_eq!(codes, ["unprotected"]);
    }
}
