//! Policy engine and destructive-action gatekeeper.
//!
//! A [`Policy`] (YAML, policy-as-code) decides what happens to a tool call:
//! **allow**, **ask** a human, or **block**. The same file gives the same
//! verdict in the local CLI and the hosted gateway.
//!
//! ```yaml
//! version: 1
//! preset: balanced          # paranoid | balanced | autopilot
//! mode: enforce             # enforce | shadow (log what would happen)
//! protected_branches: [main, master, "release/*"]
//! paths: { allow: ["/home/me/projects/**"], deny: ["**/.env"] }
//! recipients: { allow_domains: [example.com] }
//! limits: { calls_per_minute: 60, destructive_per_session: 20 }
//! rules:
//!   - name: prod-db-is-read-only
//!     match: { server: prod-postgres }
//!     when: { sql: [drop, truncate, delete_all, update_all, alter_drop] }
//!     action: block
//!   - match: { tool: "list_*" }
//!     action: allow
//! ```

pub mod glob;
pub mod limits;
pub mod risk;
pub mod shell;
pub mod sql;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use limits::{Budget, Limits};
pub use risk::Risk;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    #[default]
    Allow,
    Ask,
    Block,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Preset {
    Paranoid,
    #[default]
    Balanced,
    Autopilot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Enforce,
    Shadow,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Matcher {
    /// Glob on the server name.
    pub server: Option<String>,
    /// Glob on the tool name.
    pub tool: Option<String>,
    /// Minimum risk.
    pub risk: Option<Risk>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct When {
    /// SQL findings (`drop`, `truncate`, `delete_all`, `update_all`,
    /// `alter_drop`, `grant`, `unparsed`) or `any`.
    #[serde(default)]
    pub sql: Vec<String>,
    /// Shell finding codes (`shell.rm_rf`, `git.force_push`, ...) or `any`.
    #[serde(default)]
    pub shell: Vec<String>,
    /// Argument path → glob its string value must match, e.g. `repo: "acme/*"`.
    #[serde(default)]
    pub args: BTreeMap<String, String>,
}

impl When {
    fn is_empty(&self) -> bool {
        self.sql.is_empty() && self.shell.is_empty() && self.args.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    pub name: Option<String>,
    #[serde(rename = "match", default)]
    pub matcher: Matcher,
    #[serde(default)]
    pub when: When,
    pub action: Verdict,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathScope {
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipients {
    #[serde(default)]
    pub allow_domains: Vec<String>,
    #[serde(default)]
    pub allow: Vec<String>,
}

fn v1() -> u32 {
    1
}

fn default_protected() -> Vec<String> {
    vec![
        "main".into(),
        "master".into(),
        "production".into(),
        "prod".into(),
        "release/*".into(),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    #[serde(default = "v1")]
    pub version: u32,
    #[serde(default)]
    pub preset: Preset,
    #[serde(default)]
    pub mode: Mode,
    #[serde(default = "default_protected")]
    pub protected_branches: Vec<String>,
    #[serde(default)]
    pub paths: PathScope,
    #[serde(default)]
    pub recipients: Recipients,
    #[serde(default)]
    pub limits: Limits,
    #[serde(default)]
    pub rules: Vec<Rule>,
}

impl Default for Policy {
    fn default() -> Self {
        Policy {
            version: 1,
            preset: Preset::Balanced,
            mode: Mode::Enforce,
            protected_branches: default_protected(),
            paths: PathScope::default(),
            recipients: Recipients::default(),
            limits: Limits::default(),
            rules: Vec::new(),
        }
    }
}

impl Policy {
    pub fn from_yaml(text: &str) -> Result<Policy, String> {
        if text.trim().is_empty() {
            return Ok(Policy::default());
        }
        let p: Policy = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
        if p.version != 1 {
            return Err(format!("unsupported policy version {}", p.version));
        }
        Ok(p)
    }

    pub fn preset(preset: Preset) -> Policy {
        Policy {
            preset,
            ..Policy::default()
        }
    }
}

/// How bad an argument-level finding is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Irreversible at scale: DROP, TRUNCATE, rm -rf /, force-push to main.
    Catastrophic,
    /// Destructive but bounded: rm -rf dir, DELETE without WHERE.
    Dangerous,
    /// Outside what the policy allows: paths, recipients, protected branches.
    Scope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Reason {
    pub code: String,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<Severity>,
}

/// The policy's answer for one tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Decision {
    /// What to do now (always `allow` in shadow mode).
    pub verdict: Verdict,
    /// What enforce mode would do.
    pub would: Verdict,
    pub shadow: bool,
    pub risk: Risk,
    pub risk_source: &'static str,
    /// The explicit rule that decided, if any.
    pub rule: Option<String>,
    pub reasons: Vec<Reason>,
}

impl Decision {
    pub fn summary(&self) -> String {
        if self.reasons.is_empty() {
            return format!("{:?} tool", self.risk).to_ascii_lowercase();
        }
        self.reasons
            .iter()
            .map(|r| r.detail.as_str())
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// One tool call to judge.
pub struct Call<'a> {
    pub server: &'a str,
    pub tool: &'a str,
    /// The tool's definition from `tools/list`, if known (annotations).
    pub definition: Option<&'a Value>,
    pub arguments: &'a Value,
}

const SQL_KEYS: &[&str] = &["sql", "query", "statement", "statements", "sql_query"];
const SHELL_KEYS: &[&str] = &[
    "command",
    "cmd",
    "script",
    "shell",
    "bash",
    "commandline",
    "command_line",
];
const PATH_KEYS: &[&str] = &[
    "path",
    "paths",
    "file",
    "files",
    "filepath",
    "file_path",
    "filename",
    "directory",
    "dir",
    "source",
    "destination",
    "target",
    "src",
    "dest",
    "new_path",
    "old_path",
];
const BRANCH_KEYS: &[&str] = &["branch", "base", "ref", "target_branch", "base_branch", "head"];
const RECIPIENT_KEYS: &[&str] = &["to", "cc", "bcc", "recipient", "recipients", "email", "emails"];

fn walk<'v>(v: &'v Value, path: String, out: &mut Vec<(String, &'v str)>) {
    match v {
        Value::String(s) => out.push((path, s)),
        Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                walk(x, format!("{path}[{i}]"), out);
            }
        }
        Value::Object(m) => {
            for (k, x) in m {
                let p = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                walk(x, p, out);
            }
        }
        _ => {}
    }
}

/// The last object key in a path like `a.b[2]` → `b`.
fn leaf_key(path: &str) -> String {
    let last = path.rsplit('.').next().unwrap_or(path);
    last.split('[').next().unwrap_or(last).to_ascii_lowercase()
}

fn reason(
    code: impl Into<String>,
    detail: impl Into<String>,
    path: Option<&str>,
    severity: Option<Severity>,
) -> Reason {
    Reason {
        code: code.into(),
        detail: detail.into(),
        path: path.map(str::to_owned),
        severity,
    }
}

/// Everything risky about the arguments, independent of the policy's rules.
fn analyze_args(policy: &Policy, call: &Call<'_>) -> (Vec<Reason>, Vec<sql::SqlFinding>, Vec<shell::ShellFinding>) {
    let mut strings = Vec::new();
    walk(call.arguments, String::new(), &mut strings);
    let mut reasons = Vec::new();
    let mut sqls = Vec::new();
    let mut shells = Vec::new();
    for (path, s) in strings {
        let key = leaf_key(&path);
        if SQL_KEYS.contains(&key.as_str()) || sql::looks_like_sql(s) {
            for f in sql::analyze(s) {
                if f == sql::SqlFinding::Unparsed && !SQL_KEYS.contains(&key.as_str()) {
                    continue; // prose that merely starts with "select"
                }
                let sev = if f.catastrophic() {
                    Severity::Catastrophic
                } else {
                    Severity::Dangerous
                };
                let detail = match &f {
                    sql::SqlFinding::Drop(kind) => format!("SQL drops a {kind}"),
                    sql::SqlFinding::DropDatabase => "SQL drops a database or schema".into(),
                    sql::SqlFinding::Truncate => "SQL truncates a table".into(),
                    sql::SqlFinding::DeleteWithoutWhere => "SQL deletes every row (no WHERE)".into(),
                    sql::SqlFinding::UpdateWithoutWhere => "SQL updates every row (no WHERE)".into(),
                    sql::SqlFinding::AlterDrop => "SQL drops a column or constraint".into(),
                    sql::SqlFinding::Grant => "SQL changes permissions".into(),
                    sql::SqlFinding::Unparsed => "SQL could not be parsed".into(),
                };
                if f != sql::SqlFinding::Grant && f != sql::SqlFinding::Unparsed {
                    reasons.push(reason(f.code(), detail, Some(&path), Some(sev)));
                }
                sqls.push(f);
            }
        }
        if SHELL_KEYS.contains(&key.as_str()) {
            for f in shell::analyze(s, &policy.protected_branches) {
                let sev = if f.catastrophic {
                    Severity::Catastrophic
                } else {
                    Severity::Dangerous
                };
                reasons.push(reason(
                    f.code,
                    format!("command looks destructive ({})", f.code),
                    Some(&path),
                    Some(sev),
                ));
                shells.push(f);
            }
        }
        if PATH_KEYS.contains(&key.as_str()) {
            if s.split(['/', '\\']).any(|seg| seg == "..") {
                reasons.push(reason(
                    "path.traversal",
                    format!("path '{s}' uses '..'"),
                    Some(&path),
                    Some(Severity::Scope),
                ));
            } else if policy.paths.deny.iter().any(|g| glob::matches(g, s)) {
                reasons.push(reason(
                    "path.denied",
                    format!("path '{s}' is denied by policy"),
                    Some(&path),
                    Some(Severity::Scope),
                ));
            } else if !policy.paths.allow.is_empty() && !policy.paths.allow.iter().any(|g| glob::matches(g, s)) {
                reasons.push(reason(
                    "path.outside_scope",
                    format!("path '{s}' is outside the allowed folders"),
                    Some(&path),
                    Some(Severity::Scope),
                ));
            }
        }
        if BRANCH_KEYS.contains(&key.as_str()) {
            let branch = s.trim_start_matches("refs/heads/");
            if policy.protected_branches.iter().any(|g| glob::matches(g, branch)) {
                reasons.push(reason(
                    "git.protected_branch",
                    format!("targets protected branch '{branch}'"),
                    Some(&path),
                    Some(Severity::Scope),
                ));
            }
        }
        if RECIPIENT_KEYS.contains(&key.as_str()) {
            let r = &policy.recipients;
            if !r.allow_domains.is_empty() || !r.allow.is_empty() {
                for addr in s.split([',', ';', ' ']).map(str::trim).filter(|a| !a.is_empty()) {
                    let addr = addr.trim_matches(['<', '>']).to_ascii_lowercase();
                    let domain = addr.rsplit('@').next().unwrap_or_default();
                    let ok = r.allow.iter().any(|a| a.eq_ignore_ascii_case(&addr))
                        || r.allow_domains.iter().any(|d| d.eq_ignore_ascii_case(domain));
                    if !ok {
                        reasons.push(reason(
                            "recipient.not_allowed",
                            format!("recipient '{addr}' is not on the allow-list"),
                            Some(&path),
                            Some(Severity::Scope),
                        ));
                    }
                }
            }
        }
    }
    (reasons, sqls, shells)
}

fn rule_matches(
    rule: &Rule,
    call: &Call<'_>,
    risk: Risk,
    sqls: &[sql::SqlFinding],
    shells: &[shell::ShellFinding],
) -> bool {
    let m = &rule.matcher;
    if m.server.as_deref().is_some_and(|g| !glob::matches(g, call.server))
        || m.tool.as_deref().is_some_and(|g| !glob::matches(g, call.tool))
        || m.risk.is_some_and(|r| risk < r)
    {
        return false;
    }
    let w = &rule.when;
    if w.is_empty() {
        return true;
    }
    let sql_ok = w.sql.is_empty()
        || sqls
            .iter()
            .any(|f| w.sql.iter().any(|k| k == "any" || k == f.keyword() || k == f.code()));
    let shell_ok = w.shell.is_empty() || shells.iter().any(|f| w.shell.iter().any(|k| k == "any" || k == f.code));
    let args_ok = w.args.iter().all(|(p, g)| {
        let mut strings = Vec::new();
        walk(call.arguments, String::new(), &mut strings);
        strings.iter().any(|(path, s)| path == p && glob::matches(g, s))
    });
    sql_ok && shell_ok && args_ok
}

fn preset_verdict(preset: Preset, risk: Risk, reasons: &[Reason]) -> Verdict {
    let by_sev = |sev: Severity| -> Verdict {
        match (preset, sev) {
            (Preset::Paranoid, Severity::Catastrophic | Severity::Scope) => Verdict::Block,
            (Preset::Paranoid, Severity::Dangerous) => Verdict::Ask,
            (Preset::Balanced, _) => Verdict::Ask,
            (Preset::Autopilot, Severity::Catastrophic | Severity::Scope) => Verdict::Ask,
            (Preset::Autopilot, Severity::Dangerous) => Verdict::Allow,
        }
    };
    let by_risk = match (preset, risk) {
        (_, Risk::Read) => Verdict::Allow,
        (Preset::Paranoid, _) => Verdict::Ask,
        (Preset::Balanced, Risk::Destructive) => Verdict::Ask,
        (Preset::Balanced, _) | (Preset::Autopilot, _) => Verdict::Allow,
    };
    reasons
        .iter()
        .filter_map(|r| r.severity)
        .map(by_sev)
        .fold(by_risk, Verdict::max)
}

/// Judges one tool call.
pub fn evaluate(policy: &Policy, call: &Call<'_>) -> Decision {
    let (risk, risk_source) = risk::classify(call.tool, call.definition);
    let (mut reasons, sqls, shells) = analyze_args(policy, call);
    let explicit = policy
        .rules
        .iter()
        .find(|r| rule_matches(r, call, risk, &sqls, &shells));
    let (would, rule) = match explicit {
        Some(r) => {
            let name = r.name.clone().unwrap_or_else(|| "unnamed rule".into());
            reasons.insert(0, reason("rule", format!("policy rule '{name}'"), None, None));
            (r.action, Some(name))
        }
        None => {
            let v = preset_verdict(policy.preset, risk, &reasons);
            if v != Verdict::Allow && reasons.is_empty() {
                reasons.push(reason(
                    "risk",
                    format!("{risk:?} tool '{}'", call.tool).to_ascii_lowercase(),
                    None,
                    None,
                ));
            }
            (v, None)
        }
    };
    let shadow = policy.mode == Mode::Shadow;
    Decision {
        verdict: if shadow { Verdict::Allow } else { would },
        would,
        shadow,
        risk,
        risk_source,
        rule,
        reasons,
    }
}
