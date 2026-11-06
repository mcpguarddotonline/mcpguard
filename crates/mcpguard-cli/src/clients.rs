//! Where AI clients keep their MCP server configs, and a uniform view of them.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// `{"mcpServers": {...}}` (Claude Desktop, Claude Code, Cursor, Windsurf).
    McpServers,
    /// `{"servers": {...}}` (VS Code `mcp.json`).
    Servers,
    /// `[mcp_servers.<name>]` tables (Codex `config.toml`).
    CodexToml,
}

#[derive(Debug, Clone)]
pub struct ConfigFile {
    pub client: &'static str,
    pub path: PathBuf,
    pub format: Format,
}

/// One MCP server entry, normalized.
#[derive(Debug, Clone, PartialEq)]
pub struct Server {
    pub name: String,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub url: Option<String>,
    pub headers: Vec<(String, String)>,
}

fn home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// Known config locations on this machine (only those that exist).
pub fn discover(project: Option<&Path>) -> Vec<ConfigFile> {
    let h = home();
    let mut c = vec![
        ConfigFile {
            client: "Claude Code",
            path: h.join(".claude.json"),
            format: Format::McpServers,
        },
        ConfigFile {
            client: "Cursor",
            path: h.join(".cursor/mcp.json"),
            format: Format::McpServers,
        },
        ConfigFile {
            client: "Windsurf",
            path: h.join(".codeium/windsurf/mcp_config.json"),
            format: Format::McpServers,
        },
        ConfigFile {
            client: "Codex",
            path: h.join(".codex/config.toml"),
            format: Format::CodexToml,
        },
    ];
    if let Some(cfg) = dirs::config_dir() {
        c.push(ConfigFile {
            client: "Claude Desktop",
            path: cfg.join("Claude/claude_desktop_config.json"),
            format: Format::McpServers,
        });
        c.push(ConfigFile {
            client: "VS Code",
            path: cfg.join("Code/User/mcp.json"),
            format: Format::Servers,
        });
    }
    if let Some(p) = project {
        c.push(ConfigFile {
            client: "Project (.mcp.json)",
            path: p.join(".mcp.json"),
            format: Format::McpServers,
        });
        c.push(ConfigFile {
            client: "Project (Cursor)",
            path: p.join(".cursor/mcp.json"),
            format: Format::McpServers,
        });
        c.push(ConfigFile {
            client: "Project (VS Code)",
            path: p.join(".vscode/mcp.json"),
            format: Format::Servers,
        });
    }
    c.into_iter().filter(|f| f.path.is_file()).collect()
}

fn pairs(v: Option<&Value>) -> Vec<(String, String)> {
    v.and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                .collect()
        })
        .unwrap_or_default()
}

fn from_json(name: &str, v: &Value) -> Server {
    Server {
        name: name.to_owned(),
        command: v.get("command").and_then(Value::as_str).map(str::to_owned),
        args: v
            .get("args")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect())
            .unwrap_or_default(),
        env: pairs(v.get("env")),
        url: v
            .get("url")
            .or_else(|| v.get("serverUrl"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        headers: pairs(v.get("headers")),
    }
}

/// The servers in a config file. Claude Code's `~/.claude.json` also keeps
/// per-project servers under `projects.<path>.mcpServers`.
pub fn read(f: &ConfigFile) -> Result<Vec<Server>> {
    let text = std::fs::read_to_string(&f.path).with_context(|| format!("reading {}", f.path.display()))?;
    let mut out = Vec::new();
    match f.format {
        Format::CodexToml => {
            let t: toml::Table = toml::from_str(&text).with_context(|| format!("parsing {}", f.path.display()))?;
            if let Some(servers) = t.get("mcp_servers").and_then(|x| x.as_table()) {
                for (name, def) in servers {
                    let v: Value = serde_json::to_value(def).unwrap_or_default();
                    let mut s = from_json(name, &v);
                    s.headers.extend(pairs(v.get("http_headers")));
                    out.push(s);
                }
            }
        }
        Format::McpServers | Format::Servers => {
            let v: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", f.path.display()))?;
            let key = if f.format == Format::Servers {
                "servers"
            } else {
                "mcpServers"
            };
            if let Some(m) = v.get(key).and_then(Value::as_object) {
                out.extend(m.iter().map(|(n, d)| from_json(n, d)));
            }
            if let Some(projects) = v.get("projects").and_then(Value::as_object) {
                for (proj, pv) in projects {
                    if let Some(m) = pv.get("mcpServers").and_then(Value::as_object) {
                        let short = Path::new(proj).file_name().and_then(|x| x.to_str()).unwrap_or(proj);
                        out.extend(m.iter().map(|(n, d)| from_json(&format!("{short}/{n}"), d)));
                    }
                }
            }
        }
    }
    Ok(out)
}
