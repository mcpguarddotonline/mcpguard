//! `mcpguard wrap -- <command>`: run a stdio MCP server behind MCP-Guard.
//!
//! The client talks to us on stdin/stdout; we talk to the server as a child
//! process. Every message goes through the same engine and policy as the
//! hosted gateway. Secrets from the local vault are given to the child as
//! environment variables only (`--env GITHUB_TOKEN=vault:github`) and masked
//! back to handles in anything the server returns.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use mcpguard_engine::mcp::{self, blocked_response};
use mcpguard_engine::policy::{self, Budget, Policy, Verdict};
use mcpguard_engine::vault::MaskSet;
use serde_json::{Value, json};

use crate::vault::Store;

pub struct Options {
    pub name: String,
    pub env: Vec<(String, String)>,
    pub policy: Policy,
    pub command: Vec<String>,
}

/// Parses `NAME=vault:handle` (or `NAME=value`, passed through).
pub fn parse_env(spec: &str) -> Result<(String, String)> {
    let (k, v) = spec.split_once('=').context("--env needs NAME=vault:secret")?;
    Ok((k.to_owned(), v.to_owned()))
}

type Out = Arc<Mutex<Box<dyn Write + Send>>>;

fn send(out: &Out, v: &Value) {
    let mut o = out.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let _ = writeln!(o, "{v}");
    let _ = o.flush();
}

fn tool_error(id: &Value, text: String) -> Value {
    json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":text}],"isError":true}})
}

struct Shared {
    name: String,
    policy: Policy,
    mask: MaskSet,
    methods: Mutex<HashMap<String, String>>,
    budget: Mutex<Budget>,
    elicits: Mutex<HashMap<String, mpsc::Sender<Value>>>,
    client_elicits: Mutex<bool>,
    next: Mutex<u64>,
}

pub fn run(opts: Options, store: &dyn Store) -> Result<i32> {
    if opts.command.is_empty() {
        bail!("nothing to run: mcpguard wrap -- <command> [args]");
    }
    let mut env = Vec::new();
    let mut verifiers = Vec::new();
    let mask_key: Vec<u8> = (0..32)
        .map(|i| (i as u8).wrapping_mul(97).wrapping_add(unix_salt()))
        .collect();
    for (k, v) in &opts.env {
        match v.strip_prefix("vault:") {
            Some(secret) => {
                let value = store.get(secret)?;
                verifiers.extend(mcpguard_engine::vault::verifiers(secret, &value, &mask_key));
                env.push((k.clone(), value));
            }
            None => env.push((k.clone(), v.clone())),
        }
    }
    let mut child = Command::new(&opts.command[0])
        .args(&opts.command[1..])
        .envs(env)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("starting {}", opts.command[0]))?;
    let child_in: Out = Arc::new(Mutex::new(Box::new(child.stdin.take().context("child stdin")?)));
    let child_out = child.stdout.take().context("child stdout")?;
    let out: Out = Arc::new(Mutex::new(Box::new(std::io::stdout())));
    let sh = Arc::new(Shared {
        name: opts.name,
        policy: opts.policy,
        mask: MaskSet::new(mask_key, verifiers),
        methods: Mutex::new(HashMap::new()),
        budget: Mutex::new(Budget::default()),
        elicits: Mutex::new(HashMap::new()),
        client_elicits: Mutex::new(false),
        next: Mutex::new(1),
    });

    // Server → client.
    let (sh2, out2) = (sh.clone(), out.clone());
    let server_thread = std::thread::spawn(move || {
        for line in BufReader::new(child_out).lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(&line) {
                Ok(mut v) => {
                    from_server(&sh2, &mut v);
                    send(&out2, &v);
                }
                Err(_) => eprintln!("mcpguard: dropped a non-JSON line from the server"),
            }
        }
    });

    // Client → server.
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(mut v) = serde_json::from_str::<Value>(&line) else {
            send(
                &out,
                &json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}),
            );
            continue;
        };
        from_client(&sh, &mut v, &out, &child_in);
    }
    drop(child_in);
    let _ = server_thread.join();
    let status = child.wait()?;
    Ok(status.code().unwrap_or(1))
}

fn unix_salt() -> u8 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u8)
        .unwrap_or(7)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn from_server(sh: &Shared, v: &mut Value) {
    let method = match (v.get("method"), v.get("id")) {
        (None, Some(id)) => lock(&sh.methods).remove(&id.to_string()),
        _ => None,
    };
    sh.mask.mask_json(v);
    let r = mcp::inspect_from_server(v, method.as_deref(), &sh.name);
    if !r.findings.is_empty() {
        eprintln!(
            "mcpguard: {} {} ({})",
            format!("{:?}", r.action).to_lowercase(),
            method.unwrap_or_default(),
            r.rule_summary()
        );
    }
}

fn from_client(sh: &Arc<Shared>, v: &mut Value, out: &Out, child: &Out) {
    let id = v.get("id").cloned();
    let method = v.get("method").and_then(Value::as_str).map(str::to_owned);
    // Our own elicitation coming back.
    if method.is_none() {
        if let Some(key) = id
            .as_ref()
            .and_then(Value::as_str)
            .filter(|s| s.starts_with("mcpg-elicit-"))
        {
            if let Some(tx) = lock(&sh.elicits).remove(key) {
                let _ = tx.send(v.clone());
            }
            return;
        }
    }
    if method.as_deref() == Some("initialize") {
        *lock(&sh.client_elicits) = v.pointer("/params/capabilities/elicitation").is_some();
    }
    if let (Some(id), Some(m)) = (&id, &method) {
        lock(&sh.methods).insert(id.to_string(), m.clone());
    }
    let id = id.unwrap_or(Value::Null);
    // A real local-vault value typed by the agent: leak.
    let mut probe = v.get("params").cloned().unwrap_or(Value::Null);
    let leaked = sh.mask.mask_json(&mut probe);
    if !leaked.is_empty() {
        eprintln!("mcpguard: blocked: the agent wrote the value of {}", leaked.join(", "));
        send(
            out,
            &tool_error(
                &id,
                format!(
                    "MCP-Guard blocked this call: it contains the real value of {}. Rotate that secret.",
                    leaked.join(", ")
                ),
            ),
        );
        return;
    }
    let report = mcp::inspect_from_client(v);
    if report.blocked {
        send(out, &blocked_response(&id, method.as_deref().unwrap_or(""), &report));
        return;
    }
    if method.as_deref() == Some("tools/call") {
        let tool = v
            .pointer("/params/name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let args = v.pointer("/params/arguments").cloned().unwrap_or_else(|| json!({}));
        let d = policy::evaluate(
            &sh.policy,
            &policy::Call {
                server: &sh.name,
                tool: &tool,
                definition: None,
                arguments: &args,
            },
        );
        if let Some(limit) = lock(&sh.budget).charge(
            &sh.policy.limits,
            d.risk == policy::Risk::Destructive,
            std::time::Instant::now(),
        ) {
            send(
                out,
                &tool_error(
                    &id,
                    format!("MCP-Guard stopped this call: session limit reached ({limit})."),
                ),
            );
            return;
        }
        match d.verdict {
            Verdict::Allow => {}
            Verdict::Block => {
                send(
                    out,
                    &tool_error(&id, format!("MCP-Guard policy blocked {tool}: {}.", d.summary())),
                );
                return;
            }
            Verdict::Ask => {
                if !(*lock(&sh.client_elicits) && sh.policy.approvals.in_client) {
                    send(
                        out,
                        &tool_error(
                            &id,
                            format!(
                                "MCP-Guard: {tool} needs approval ({}), and this client can't show approval prompts. Change the policy to allow it.",
                                d.summary()
                            ),
                        ),
                    );
                    return;
                }
                // Ask in the client without blocking the read loop.
                let (sh, out, child, msg) = (sh.clone(), out.clone(), child.clone(), v.clone());
                std::thread::spawn(move || ask(&sh, &out, &child, msg, id, &tool, &d.summary()));
                return;
            }
        }
    }
    send(child, v);
}

fn ask(sh: &Shared, out: &Out, child: &Out, msg: Value, id: Value, tool: &str, why: &str) {
    let eid = {
        let mut n = lock(&sh.next);
        *n += 1;
        format!("mcpg-elicit-{}", *n)
    };
    let (tx, rx) = mpsc::channel();
    lock(&sh.elicits).insert(eid.clone(), tx);
    let args = msg
        .pointer("/params/arguments")
        .map(Value::to_string)
        .unwrap_or_default();
    send(
        out,
        &json!({
            "jsonrpc": "2.0", "id": eid, "method": "elicitation/create",
            "params": {
                "message": format!("MCP-Guard: the agent wants to run {tool} ({why}). Allow it?\n\nArguments: {}", args.chars().take(600).collect::<String>()),
                "requestedSchema": {"type": "object", "properties": {"decision": {"type": "string", "enum": ["approve", "deny"], "enumNames": ["Approve", "Deny"]}}, "required": ["decision"]}
            }
        }),
    );
    let timeout = Duration::from_secs(u64::from(sh.policy.approvals.timeout_seconds));
    let approved = match rx.recv_timeout(timeout) {
        Ok(a) => {
            a.pointer("/result/action").and_then(Value::as_str) == Some("accept")
                && a.pointer("/result/content/decision").and_then(Value::as_str) == Some("approve")
        }
        Err(_) => false,
    };
    lock(&sh.elicits).remove(&eid);
    if approved {
        send(child, &msg);
    } else {
        send(
            out,
            &tool_error(&id, format!("{tool} was not approved, so it was not run.")),
        );
    }
}
