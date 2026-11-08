//! `mcpguard login`, `logout`, `sync`: connect the local CLI to a hosted
//! MCP-Guard workspace.
//!
//! Login is OAuth 2.1 for native apps (RFC 8252): discover the gateway's
//! authorization server (RFC 9728, RFC 8414), register a client for this
//! device (RFC 7591), and run the authorization-code flow with PKCE through a
//! one-shot loopback listener. The CLI then shows up as an agent in the
//! workspace, revocable from the dashboard.
//!
//! Tokens live in the OS keychain; `account.json` holds only URLs and ids.
//! Sync pulls the workspace policy into `policy.yaml` (what `wrap` reads) and
//! pushes verdict metadata spooled by `wrap`: names, actions and rule ids,
//! never prompts, arguments or results.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::vault::config_dir;

const KEYRING_SERVICE: &str = "mcpguard-cloud";
const SPOOL_MAX_BYTES: u64 = 5 * 1024 * 1024;
const BATCH: usize = 500;

#[derive(Serialize, Deserialize, Clone)]
pub struct Account {
    pub gateway: String,
    pub issuer: String,
    pub client_id: String,
    pub token_endpoint: String,
    pub revocation_endpoint: Option<String>,
    /// Unix seconds when the cached access token expires.
    #[serde(default)]
    pub access_expires_at: u64,
}

fn account_path() -> PathBuf {
    config_dir().join("account.json")
}

pub fn spool_path() -> PathBuf {
    config_dir().join("events.jsonl")
}

pub fn policy_path() -> PathBuf {
    config_dir().join("policy.yaml")
}

pub fn load() -> Option<Account> {
    std::fs::read_to_string(account_path())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
}

fn save(a: &Account) -> Result<()> {
    std::fs::create_dir_all(config_dir())?;
    std::fs::write(account_path(), serde_json::to_vec_pretty(a)?).context("writing account.json")
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn secret(name: &str) -> Result<keyring::Entry> {
    keyring::Entry::new(KEYRING_SERVICE, name).context("opening the OS keychain")
}

fn random_b64(n: usize) -> Result<String> {
    let mut b = vec![0u8; n];
    getrandom::fill(&mut b).map_err(|e| anyhow!("OS random generator: {e}"))?;
    Ok(B64.encode(b))
}

fn get_json(url: &str) -> Result<Value> {
    ureq::get(url)
        .header("Accept", "application/json")
        .call()
        .with_context(|| format!("GET {url}"))?
        .body_mut()
        .read_json()
        .with_context(|| format!("reading {url}"))
}

fn post_form(url: &str, form: &[(&str, &str)]) -> Result<Value> {
    let mut res = ureq::post(url)
        .config()
        .http_status_as_error(false)
        .build()
        .send_form(form.iter().copied())
        .with_context(|| format!("POST {url}"))?;
    let status = res.status().as_u16();
    let body: Value = res.body_mut().read_json().unwrap_or(Value::Null);
    if !(200..300).contains(&status) {
        let msg = body["error_description"]
            .as_str()
            .or(body["message"].as_str())
            .or(body["error"].as_str())
            .unwrap_or("request failed");
        bail!("{url} answered {status}: {msg}");
    }
    Ok(body)
}

/// Strips a trailing `/mcp` and slashes: `https://gw.example/mcp` → `https://gw.example`.
fn gateway_base(gateway: &str) -> String {
    gateway
        .trim_end_matches('/')
        .trim_end_matches("/mcp")
        .trim_end_matches('/')
        .to_owned()
}

pub fn pkce_challenge(verifier: &str) -> String {
    B64.encode(Sha256::digest(verifier.as_bytes()))
}

fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(url).status();
    #[cfg(target_os = "windows")]
    let r = std::process::Command::new("cmd")
        .args(["/C", "start", "", url])
        .status();
    #[cfg(all(unix, not(target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(url).status();
    if r.map(|s| !s.success()).unwrap_or(true) {
        eprintln!("Couldn't open a browser. Open the link above yourself.");
    }
}

/// Decodes `application/x-www-form-urlencoded` (good enough for OAuth params).
fn decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => {
                match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16) {
                    Ok(v) => {
                        out.push(v);
                        i += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn encode(s: &str) -> String {
    s.bytes()
        .map(|c| match c {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (c as char).to_string(),
            _ => format!("%{c:02X}"),
        })
        .collect()
}

pub fn query_params(target: &str) -> Vec<(String, String)> {
    target
        .split_once('?')
        .map(|(_, q)| q)
        .unwrap_or("")
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (decode(k), decode(v))
        })
        .collect()
}

/// Waits for the browser to come back to `/callback`; returns its query params.
fn wait_for_callback(listener: TcpListener, timeout: Duration) -> Result<Vec<(String, String)>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut line = String::new();
            if BufReader::new(&stream).read_line(&mut line).is_err() {
                continue;
            }
            let target = line.split_whitespace().nth(1).unwrap_or("").to_owned();
            if !target.starts_with("/callback") {
                let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                continue;
            }
            let params = query_params(&target);
            let ok = params.iter().any(|(k, _)| k == "code");
            let body = if ok {
                "<!doctype html><meta charset=utf-8><title>MCP-Guard</title><body style=\"font:16px -apple-system,system-ui,sans-serif;display:grid;place-items:center;height:90vh\"><div><h1 style=\"font-size:22px\">You're signed in.</h1><p>You can close this tab and go back to your terminal.</p></div>"
            } else {
                "<!doctype html><meta charset=utf-8><title>MCP-Guard</title><body style=\"font:16px -apple-system,system-ui,sans-serif;display:grid;place-items:center;height:90vh\"><div><h1 style=\"font-size:22px\">Sign-in didn't finish.</h1><p>Go back to your terminal for details.</p></div>"
            };
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = tx.send(params);
            break;
        }
    });
    rx.recv_timeout(timeout)
        .map_err(|_| anyhow!("timed out waiting for the browser (5 minutes)"))
}

pub fn login(gateway: &str, open: bool) -> Result<i32> {
    let base = gateway_base(gateway);
    let prm =
        get_json(&format!("{base}/.well-known/oauth-protected-resource")).context("is this an MCP-Guard gateway?")?;
    let resource = prm["resource"]
        .as_str()
        .context("gateway metadata has no resource")?
        .to_owned();
    let issuer = prm["authorization_servers"][0]
        .as_str()
        .context("gateway metadata names no authorization server")?
        .trim_end_matches('/')
        .to_owned();
    let meta = get_json(&format!("{issuer}/.well-known/oauth-authorization-server"))?;
    let field = |k: &str| {
        meta[k]
            .as_str()
            .map(str::to_owned)
            .with_context(|| format!("authorization server has no {k}"))
    };
    let (authorize, token, register) = (
        field("authorization_endpoint")?,
        field("token_endpoint")?,
        field("registration_endpoint")?,
    );

    let listener = TcpListener::bind("127.0.0.1:0").context("opening a local port for the sign-in callback")?;
    let redirect = format!("http://127.0.0.1:{}/callback", listener.local_addr()?.port());
    let host = std::env::var("COMPUTERNAME")
        .ok()
        .or_else(|| {
            std::process::Command::new("hostname")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|h| h.trim().trim_end_matches(".local").to_owned())
        })
        .filter(|h| !h.is_empty() && h.len() <= 64)
        .unwrap_or_else(|| "this computer".into());
    let mut res = ureq::post(&register)
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(json!({
            "client_name": format!("mcpguard CLI on {host}"),
            "redirect_uris": [redirect],
            "grant_types": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_method": "none",
        }))
        .context("registering this device")?;
    let reg: Value = res.body_mut().read_json().unwrap_or(Value::Null);
    let client_id = reg["client_id"]
        .as_str()
        .context("registration returned no client_id")?
        .to_owned();

    let verifier = random_b64(32)?;
    let state = random_b64(16)?;
    let url = format!(
        "{authorize}?response_type=code&client_id={}&redirect_uri={}&code_challenge={}&code_challenge_method=S256&state={}&scope=mcp&resource={}",
        encode(&client_id),
        encode(&redirect),
        encode(&pkce_challenge(&verifier)),
        encode(&state),
        encode(&resource),
    );
    println!("Sign in to MCP-Guard in your browser:\n\n  {url}\n");
    if open {
        open_browser(&url);
    }
    let params = wait_for_callback(listener, Duration::from_secs(300))?;
    let get = |k: &str| params.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
    if get("state").as_deref() != Some(state.as_str()) {
        bail!("the sign-in response didn't match this request; try again");
    }
    if let Some(e) = get("error") {
        bail!("sign-in was not completed: {}", get("error_description").unwrap_or(e));
    }
    let code = get("code").context("the browser came back without a code")?;
    let tok = post_form(
        &token,
        &[
            ("grant_type", "authorization_code"),
            ("code", &code),
            ("redirect_uri", &redirect),
            ("client_id", &client_id),
            ("code_verifier", &verifier),
            ("resource", &resource),
        ],
    )?;
    let mut acct = Account {
        gateway: resource,
        issuer,
        client_id,
        token_endpoint: token,
        revocation_endpoint: meta["revocation_endpoint"].as_str().map(str::to_owned),
        access_expires_at: 0,
    };
    if let Err(e) = store_tokens(&mut acct, &tok).and_then(|()| save(&acct)) {
        // Don't leave a live grant behind that nothing on this device can use or revoke.
        if let Some(url) = &acct.revocation_endpoint {
            for k in ["refresh_token", "access_token"] {
                if let Some(t) = tok[k].as_str() {
                    let _ = post_form(
                        url,
                        &[("token", t), ("token_type_hint", k), ("client_id", &acct.client_id)],
                    );
                }
            }
        }
        return Err(e);
    }
    println!("Signed in. This computer now appears under Agents in your dashboard.");
    sync()?;
    Ok(0)
}

fn store_tokens(acct: &mut Account, tok: &Value) -> Result<()> {
    let access = tok["access_token"].as_str().context("no access_token in response")?;
    secret("access_token")?
        .set_password(access)
        .context("saving to the OS keychain")?;
    if let Some(r) = tok["refresh_token"].as_str() {
        secret("refresh_token")?
            .set_password(r)
            .context("saving to the OS keychain")?;
    }
    let ttl = tok["expires_in"].as_u64().unwrap_or(3600);
    acct.access_expires_at = now() + ttl.saturating_sub(60);
    Ok(())
}

/// A valid access token, refreshing (with rotation) when it's about to expire.
fn access_token(acct: &mut Account) -> Result<String> {
    if acct.access_expires_at > now()
        && let Ok(t) = secret("access_token")?.get_password()
    {
        return Ok(t);
    }
    let refresh = secret("refresh_token")?
        .get_password()
        .map_err(|_| anyhow!("not signed in: run `mcpguard login`"))?;
    let tok = post_form(
        &acct.token_endpoint,
        &[
            ("grant_type", "refresh_token"),
            ("refresh_token", &refresh),
            ("client_id", &acct.client_id),
        ],
    )
    .context("refreshing the sign-in (if this keeps failing, run `mcpguard login` again)")?;
    store_tokens(acct, &tok)?;
    save(acct)?;
    secret("access_token")?
        .get_password()
        .context("reading the OS keychain")
}

pub fn logout() -> Result<i32> {
    let Some(acct) = load() else {
        println!("Not signed in.");
        return Ok(0);
    };
    if let Some(url) = &acct.revocation_endpoint {
        // Best effort: revoke both, so nothing stays valid until it expires.
        for name in ["refresh_token", "access_token"] {
            if let Ok(t) = secret(name)?.get_password() {
                let _ = post_form(
                    url,
                    &[("token", &t), ("token_type_hint", name), ("client_id", &acct.client_id)],
                );
            }
        }
    }
    for n in ["access_token", "refresh_token"] {
        let _ = secret(n)?.delete_credential();
    }
    std::fs::remove_file(account_path()).ok();
    println!("Signed out. Your synced policy stays at {}.", policy_path().display());
    Ok(0)
}

/// Pulls the workspace policy, then pushes spooled verdict metadata.
pub fn sync() -> Result<i32> {
    let mut acct = load().context("not signed in: run `mcpguard login`")?;
    let base = gateway_base(&acct.gateway);
    let token = access_token(&mut acct)?;

    let p: Value = ureq::get(&format!("{base}/v1/policy"))
        .header("Authorization", &format!("Bearer {token}"))
        .call()
        .context("downloading the workspace policy")?
        .body_mut()
        .read_json()?;
    if p["is_default"].as_bool() == Some(true) {
        println!("Policy: the workspace uses the default; local wrap uses the balanced preset.");
    } else {
        let yaml = p["yaml"].as_str().unwrap_or_default();
        mcpguard_engine::policy::Policy::from_yaml(yaml)
            .map_err(|e| anyhow!("the workspace policy doesn't parse with this CLI version ({e}); update mcpguard"))?;
        let path = policy_path();
        if std::fs::read_to_string(&path).ok().as_deref() != Some(yaml) {
            std::fs::create_dir_all(config_dir())?;
            if path.exists() {
                std::fs::copy(&path, path.with_extension("yaml.before-sync")).ok();
            }
            std::fs::write(&path, yaml)?;
            println!("Policy: updated {}", path.display());
        } else {
            println!("Policy: up to date");
        }
    }

    let sent = push_spool(&base, &token)?;
    println!(
        "Activity: uploaded {sent} local decision{}",
        if sent == 1 { "" } else { "s" }
    );
    Ok(0)
}

fn push_spool(base: &str, token: &str) -> Result<usize> {
    let spool = spool_path();
    if !spool.exists() {
        return Ok(0);
    }
    // Move the spool aside first so `wrap` keeps appending to a fresh file.
    let sending = spool.with_extension("jsonl.sending");
    if !sending.exists() {
        std::fs::rename(&spool, &sending)?;
    }
    let text = std::fs::read_to_string(&sending)?;
    let events: Vec<Value> = text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    let mut sent = 0;
    for chunk in events.chunks(BATCH) {
        let r: Value = ureq::post(&format!("{base}/v1/events"))
            .header("Authorization", &format!("Bearer {token}"))
            .send_json(json!({ "events": chunk }))
            .context("uploading activity")?
            .body_mut()
            .read_json()?;
        sent += r["accepted"].as_u64().unwrap_or(0) as usize;
    }
    std::fs::remove_file(&sending)?;
    Ok(sent)
}

/// Called by `wrap` for every non-allow decision while signed in. Metadata only.
pub fn spool(server: &str, tool: Option<&str>, direction: &str, action: &str, rules: &[String]) {
    if !account_path().exists() {
        return;
    }
    let path = spool_path();
    if std::fs::metadata(&path)
        .map(|m| m.len() > SPOOL_MAX_BYTES)
        .unwrap_or(false)
    {
        return; // never grow without bound if sync isn't running
    }
    let line = json!({
        "server": server, "tool": tool, "direction": direction, "action": action,
        "rules": rules.iter().take(20).collect::<Vec<_>>(), "at": now(),
    });
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{line}");
    }
}

pub fn status() -> Result<i32> {
    match load() {
        Some(a) => {
            println!("Signed in to {} (gateway {})", a.issuer, a.gateway);
            let pending = std::fs::read_to_string(spool_path())
                .map(|t| t.lines().count())
                .unwrap_or(0);
            println!("Local decisions waiting to sync: {pending}");
        }
        None => println!("Not signed in. Run `mcpguard login` to connect a hosted workspace."),
    }
    println!("Policy file: {}", policy_path().display());
    println!("Config folder: {}", config_dir().display());
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_rfc7636_appendix_b() {
        assert_eq!(
            pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn parses_callback_query() {
        let p = query_params("/callback?code=a%2Bb%20c&state=x+y&empty=");
        assert_eq!(
            p,
            vec![
                ("code".into(), "a+b c".into()),
                ("state".into(), "x y".into()),
                ("empty".into(), String::new())
            ]
        );
        assert_eq!(encode("http://127.0.0.1:5/cb"), "http%3A%2F%2F127.0.0.1%3A5%2Fcb");
        assert_eq!(
            gateway_base("https://gw.mcpguard.online/mcp/"),
            "https://gw.mcpguard.online"
        );
    }
}
