//! `mcpguard`: local, open-source MCP firewall.
//!
//! - `scan`: audit the MCP configs of Claude, Cursor, VS Code, Codex, Windsurf.
//! - `wrap`: run a stdio MCP server behind the engine, policy and local vault.
//! - `install` / `uninstall`: wrap every server, move secrets into the vault,
//!   add the prompt hook; one-command undo.
//! - `vault`: secrets in the OS keychain.
//! - `login` / `sync`: connect to a hosted workspace (policy down, metadata up).
//! - `policy`, `inspect`, `hook`: the building blocks.

mod account;
mod clients;
mod hook;
mod install;
mod scan;
mod vault;
mod wrap;

use std::io::{Read, Write};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use mcpguard_engine::{Action, Direction, inspect_bytes};

#[derive(Parser)]
#[command(
    name = "mcpguard",
    version,
    about = "Firewall for AI agents' MCP tools: blocks prompt injection, secret leaks and destructive actions."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Inspect text (stdin or --text) and print the verdict as JSON.
    ///
    /// Exit code: 0 = allow, 2 = findings (annotate/redact/ask), 3 = block.
    Inspect {
        /// Which way the text travels: into the model (tool results) or out to a server (tool arguments).
        #[arg(long, value_enum, default_value_t = Dir::ToClient)]
        direction: Dir,
        /// Text to inspect; reads stdin when omitted.
        #[arg(long)]
        text: Option<String>,
        /// Print the forwarded (redacted/normalized) text instead of JSON.
        #[arg(long)]
        output: bool,
    },
    /// Work with policy files (the same file the hosted gateway uses).
    Policy {
        #[command(subcommand)]
        cmd: PolicyCmd,
    },
    /// Audit the MCP configs of Claude Desktop/Code, Cursor, VS Code, Codex and Windsurf.
    ///
    /// Exit code 2 when a high-risk issue (e.g. a plaintext token) is found.
    Scan {
        /// Also check this project's .mcp.json / .cursor / .vscode configs.
        #[arg(long)]
        project: Option<std::path::PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Run a stdio MCP server behind MCP-Guard: `mcpguard wrap --env GITHUB_TOKEN=vault:github -- npx server-github`.
    Wrap {
        /// Server name used in policies and messages.
        #[arg(long)]
        name: Option<String>,
        /// Environment for the server; `NAME=vault:secret` reads the local vault.
        #[arg(long = "env", value_name = "NAME=vault:SECRET")]
        env: Vec<String>,
        /// Policy file (default: policy.yaml in the config folder, see `mcpguard status`; else balanced).
        #[arg(long)]
        policy: Option<std::path::PathBuf>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
        command: Vec<String>,
    },
    /// Wrap every stdio server in your clients, move plaintext secrets into the vault, add the prompt hook.
    Install {
        #[arg(long)]
        project: Option<std::path::PathBuf>,
        /// Show what would change without changing anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Restore the client configs `install` changed.
    Uninstall,
    /// Local vault in the OS keychain.
    Vault {
        #[command(subcommand)]
        cmd: VaultCmd,
    },
    /// Hooks for AI clients (Claude Code `UserPromptSubmit`: `mcpguard hook prompt`).
    Hook {
        #[command(subcommand)]
        cmd: HookCmd,
    },
    /// Connect this computer to a hosted MCP-Guard workspace (opens your browser).
    Login {
        /// Gateway to sign in to.
        #[arg(long, default_value = "https://gw.mcpguard.online")]
        gateway: String,
        /// Print the sign-in link instead of opening a browser.
        #[arg(long)]
        no_browser: bool,
    },
    /// Sign out and revoke this computer's access.
    Logout,
    /// Download the workspace policy and upload local decisions (metadata only).
    Sync,
    /// Show whether this computer is connected to a workspace, and where files live.
    Status {
        /// Print only the policy file path `wrap` reads by default.
        #[arg(long)]
        policy_path: bool,
    },
}

#[derive(Subcommand)]
enum VaultCmd {
    /// Store a secret (value read from stdin, or prompted without echo where supported).
    Add { name: String },
    /// List secret names (never values).
    List,
    /// Remove a secret.
    Rm { name: String },
}

#[derive(Subcommand)]
enum HookCmd {
    /// Block prompts that contain secrets (reads the hook JSON on stdin; exit 2 = block).
    Prompt,
}

#[derive(Subcommand)]
enum PolicyCmd {
    /// Show what a policy decides for one tool call. Exit code: 0 allow, 2 ask, 3 block.
    Check {
        /// Policy file (YAML). Defaults to the built-in balanced preset.
        #[arg(long)]
        policy: Option<std::path::PathBuf>,
        #[arg(long)]
        server: String,
        #[arg(long)]
        tool: String,
        /// Tool arguments as JSON.
        #[arg(long, default_value = "{}")]
        args: String,
    },
    /// Check that a policy file is valid.
    Validate { file: std::path::PathBuf },
    /// Print a starter policy.
    Init,
    /// Hash a term for `detectors.never_leave` (the policy stores only the hash).
    HashTerm {
        #[arg(long)]
        salt: String,
        term: String,
    },
}

const STARTER_POLICY: &str = r#"# MCP-Guard policy. Docs: https://mcpguard.online/docs/policies
version: 1
preset: balanced          # paranoid | balanced | autopilot
mode: enforce             # shadow = log what would be blocked, block nothing
protected_branches: [main, master, production, "release/*"]
# paths: { allow: ["/home/me/projects/**"], deny: ["**/.env", "**/.ssh/**"] }
# recipients: { allow_domains: [example.com] }
limits: { calls_per_minute: 120, destructive_per_session: 25 }
rules:
  # - name: prod-db-read-only
  #   match: { server: "prod-*" }
  #   when: { sql: [drop, truncate, delete_all, update_all, alter_drop] }
  #   action: block
"#;

#[derive(Clone, Copy, ValueEnum)]
enum Dir {
    ToClient,
    ToServer,
}

impl From<Dir> for Direction {
    fn from(d: Dir) -> Self {
        match d {
            Dir::ToClient => Direction::ToClient,
            Dir::ToServer => Direction::ToServer,
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("mcpguard: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.cmd {
        Cmd::Inspect {
            direction,
            text,
            output,
        } => {
            let bytes = match text {
                Some(t) => t.into_bytes(),
                None => {
                    let mut buf = Vec::new();
                    std::io::stdin().read_to_end(&mut buf).context("reading stdin")?;
                    buf
                }
            };
            let v = inspect_bytes(&bytes, direction.into());
            let mut out = std::io::stdout().lock();
            if output {
                out.write_all(v.output.as_bytes())?;
            } else {
                serde_json::to_writer_pretty(&mut out, &v)?;
                writeln!(out)?;
            }
            Ok(ExitCode::from(match v.action {
                Action::Allow => 0,
                Action::Block => 3,
                _ => 2,
            }))
        }
        Cmd::Policy { cmd } => policy(cmd),
        Cmd::Scan { project, json } => Ok(code(scan::run(project.as_deref(), json)?)),
        Cmd::Wrap {
            name,
            env,
            policy,
            command,
        } => {
            let policy = match policy {
                Some(p) => load_policy(&p)?,
                None => {
                    let default = vault::config_dir().join("policy.yaml");
                    if default.is_file() {
                        load_policy(&default)?
                    } else {
                        mcpguard_engine::policy::Policy::default()
                    }
                }
            };
            let name = name.unwrap_or_else(|| {
                std::path::Path::new(&command[0])
                    .file_name()
                    .and_then(|x| x.to_str())
                    .unwrap_or("server")
                    .to_owned()
            });
            let env = env.iter().map(|e| wrap::parse_env(e)).collect::<Result<Vec<_>>>()?;
            let store = vault::Keychain::open();
            Ok(code(wrap::run(
                wrap::Options {
                    name,
                    env,
                    policy,
                    command,
                },
                &store,
            )?))
        }
        Cmd::Install { project, dry_run } => {
            let mut store = vault::Keychain::open();
            Ok(code(install::install(project.as_deref(), dry_run, &mut store)?))
        }
        Cmd::Uninstall => Ok(code(install::uninstall()?)),
        Cmd::Vault { cmd } => vault_cmd(cmd),
        Cmd::Hook { cmd: HookCmd::Prompt } => Ok(code(hook::prompt(&vault::Keychain::open())?)),
        Cmd::Login { gateway, no_browser } => Ok(code(account::login(&gateway, !no_browser)?)),
        Cmd::Logout => Ok(code(account::logout()?)),
        Cmd::Sync => Ok(code(account::sync()?)),
        Cmd::Status { policy_path: true } => {
            println!("{}", account::policy_path().display());
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Status { policy_path: false } => Ok(code(account::status()?)),
    }
}

fn load_policy(path: &std::path::Path) -> Result<mcpguard_engine::policy::Policy> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    mcpguard_engine::policy::Policy::from_yaml(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

fn policy(cmd: PolicyCmd) -> Result<ExitCode> {
    use mcpguard_engine::policy::{Call, Policy, Verdict, evaluate};
    match cmd {
        PolicyCmd::Init => {
            print!("{STARTER_POLICY}");
            Ok(ExitCode::SUCCESS)
        }
        PolicyCmd::HashTerm { salt, term } => {
            println!("{}", mcpguard_engine::policy::detect::hash_term(&salt, &term));
            Ok(ExitCode::SUCCESS)
        }
        PolicyCmd::Validate { file } => {
            let p = load_policy(&file)?;
            println!(
                "ok: preset {:?}, mode {:?}, {} rule(s)",
                p.preset,
                p.mode,
                p.rules.len()
            );
            Ok(ExitCode::SUCCESS)
        }
        PolicyCmd::Check {
            policy,
            server,
            tool,
            args,
        } => {
            let p = match policy {
                Some(f) => load_policy(&f)?,
                None => Policy::default(),
            };
            let arguments: serde_json::Value = serde_json::from_str(&args).context("--args must be JSON")?;
            let d = evaluate(
                &p,
                &Call {
                    server: &server,
                    tool: &tool,
                    definition: None,
                    arguments: &arguments,
                },
            );
            let mut out = std::io::stdout().lock();
            serde_json::to_writer_pretty(&mut out, &d)?;
            writeln!(out)?;
            Ok(ExitCode::from(match d.verdict {
                Verdict::Allow => 0,
                Verdict::Ask => 2,
                Verdict::Block => 3,
            }))
        }
    }
}

fn code(c: i32) -> ExitCode {
    ExitCode::from(u8::try_from(c).unwrap_or(1))
}

fn vault_cmd(cmd: VaultCmd) -> Result<ExitCode> {
    use vault::Store;
    let mut store = vault::Keychain::open();
    match cmd {
        VaultCmd::Add { name } => {
            eprint!("Value for {name} (input is read from stdin; paste and press Enter): ");
            let mut value = String::new();
            std::io::stdin().read_line(&mut value)?;
            let value = value.trim_end_matches(['\r', '\n']);
            anyhow::ensure!(!value.is_empty(), "empty value");
            store.set(&name, value)?;
            eprintln!();
            println!("Stored {name}. Use it with: mcpguard wrap --env VAR=vault:{name} -- <server>");
        }
        VaultCmd::List => {
            for n in store.names()? {
                println!("{n}");
            }
        }
        VaultCmd::Rm { name } => {
            store.remove(&name)?;
            println!("Removed {name}.");
        }
    }
    Ok(ExitCode::SUCCESS)
}
