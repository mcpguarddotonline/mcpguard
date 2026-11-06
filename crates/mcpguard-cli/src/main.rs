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
        Cmd::Scan => bail!("`scan` is not implemented yet — follow https://mcpguard.online"),
        Cmd::Wrap { .. } => bail!("`wrap` is not implemented yet — follow https://mcpguard.online"),
        Cmd::Install => bail!("`install` is not implemented yet — follow https://mcpguard.online"),
    }
}
