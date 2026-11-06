//! `mcpguard` — local, open-source MCP firewall.
//!
//! Today: `inspect` runs the detection engine on text. Coming next: `scan`
//! (audit your AI clients' MCP configs), `wrap` (guard a stdio MCP server) and
//! `install` (route clients through MCP-Guard, with one-command undo).

use std::io::{Read, Write};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
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
    /// Audit the MCP configs of Claude, Cursor, VS Code, Codex and Windsurf (coming soon).
    Scan,
    /// Run a stdio MCP server behind MCP-Guard (coming soon): `mcpguard wrap -- npx some-server`.
    Wrap {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
        command: Vec<String>,
    },
    /// Route your AI clients through MCP-Guard, with backup and undo (coming soon).
    Install,
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
        Cmd::Scan => bail!("`scan` is not implemented yet — follow https://mcpguard.online"),
        Cmd::Wrap { .. } => bail!("`wrap` is not implemented yet — follow https://mcpguard.online"),
        Cmd::Install => bail!("`install` is not implemented yet — follow https://mcpguard.online"),
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
