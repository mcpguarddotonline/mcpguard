//! Policy engine behaviour (the same file gives the same verdict everywhere).

use mcpguard_engine::policy::{Call, Policy, Preset, Risk, Verdict, evaluate};
use serde_json::{Value, json};

fn judge(policy: &Policy, server: &str, tool: &str, args: Value) -> mcpguard_engine::policy::Decision {
    evaluate(
        policy,
        &Call {
            server,
            tool,
            definition: None,
            arguments: &args,
        },
    )
}

#[test]
fn drop_table_is_held_for_approval_under_balanced() {
    let p = Policy::default();
    let d = judge(&p, "postgres", "query", json!({"sql": "DROP TABLE customers"}));
    assert_eq!(d.verdict, Verdict::Ask);
    assert_eq!(d.reasons[0].code, "sql.drop");
    // Reads go straight through.
    let d = judge(
        &p,
        "postgres",
        "query",
        json!({"sql": "SELECT * FROM customers WHERE id = 1"}),
    );
    assert_eq!(d.verdict, Verdict::Allow);
}

#[test]
fn presets_differ_on_the_same_call() {
    let args = json!({"command": "rm -rf build"});
    assert_eq!(
        judge(&Policy::preset(Preset::Paranoid), "sh", "run_command", args.clone()).verdict,
        Verdict::Ask
    );
    assert_eq!(
        judge(&Policy::preset(Preset::Balanced), "sh", "run_command", args.clone()).verdict,
        Verdict::Ask
    );
    assert_eq!(
        judge(&Policy::preset(Preset::Autopilot), "sh", "run_command", args).verdict,
        Verdict::Allow
    );
    let nuke = json!({"command": "rm -rf /"});
    assert_eq!(
        judge(&Policy::preset(Preset::Paranoid), "sh", "run_command", nuke.clone()).verdict,
        Verdict::Block
    );
    assert_eq!(
        judge(&Policy::preset(Preset::Autopilot), "sh", "run_command", nuke).verdict,
        Verdict::Ask
    );
    // Paranoid asks before any write; balanced doesn't.
    assert_eq!(
        judge(&Policy::preset(Preset::Paranoid), "gh", "create_issue", json!({})).verdict,
        Verdict::Ask
    );
    assert_eq!(
        judge(&Policy::default(), "gh", "create_issue", json!({})).verdict,
        Verdict::Allow
    );
    assert_eq!(
        judge(&Policy::default(), "gh", "delete_repository", json!({"repo": "x"})).risk,
        Risk::Destructive
    );
}

#[test]
fn yaml_rules_override_presets() {
    let p = Policy::from_yaml(
        r#"
preset: autopilot
rules:
  - name: prod-db-read-only
    match: { server: "prod-*" }
    when: { sql: [drop, truncate, delete_all] }
    action: block
  - name: trust-staging
    match: { server: staging, risk: destructive }
    action: allow
  - match: { tool: "merge_pull_request" }
    when: { args: { base: main } }
    action: ask
"#,
    )
    .expect("policy");
    assert_eq!(
        judge(&p, "prod-pg", "query", json!({"sql": "TRUNCATE t"})).verdict,
        Verdict::Block
    );
    assert_eq!(
        judge(&p, "prod-pg", "query", json!({"sql": "SELECT 1"})).verdict,
        Verdict::Allow
    );
    let d = judge(&p, "staging", "delete_branch", json!({"branch": "main"}));
    assert_eq!((d.verdict, d.rule.as_deref()), (Verdict::Allow, Some("trust-staging")));
    assert_eq!(
        judge(&p, "gh", "merge_pull_request", json!({"base": "main"})).verdict,
        Verdict::Ask
    );
}

#[test]
fn shadow_mode_reports_but_allows() {
    let p = Policy::from_yaml("mode: shadow\npreset: paranoid\n").expect("policy");
    let d = judge(&p, "sh", "run_command", json!({"command": "rm -rf /"}));
    assert_eq!(d.verdict, Verdict::Allow);
    assert_eq!(d.would, Verdict::Block);
    assert!(d.shadow);
}

#[test]
fn paths_recipients_and_branches() {
    let p = Policy::from_yaml(
        r#"
paths: { allow: ["/work/**"], deny: ["**/.env"] }
recipients: { allow_domains: [acme.com] }
"#,
    )
    .expect("policy");
    assert_eq!(
        judge(&p, "fs", "write_file", json!({"path": "/work/a.txt"})).verdict,
        Verdict::Allow
    );
    let d = judge(&p, "fs", "write_file", json!({"path": "/etc/passwd"}));
    assert_eq!(
        (d.verdict, d.reasons[0].code.as_str()),
        (Verdict::Ask, "path.outside_scope")
    );
    assert_eq!(
        judge(&p, "fs", "read_file", json!({"path": "/work/../etc/shadow"})).reasons[0].code,
        "path.traversal"
    );
    assert_eq!(
        judge(&p, "fs", "read_file", json!({"path": "/work/app/.env"})).reasons[0].code,
        "path.denied"
    );
    assert_eq!(
        judge(&p, "mail", "send_email", json!({"to": "bob@acme.com"})).verdict,
        Verdict::Allow
    );
    assert_eq!(
        judge(&p, "mail", "send_email", json!({"to": "eve@evil.example"})).verdict,
        Verdict::Ask
    );
    assert_eq!(
        judge(&p, "gh", "push_files", json!({"branch": "main"})).reasons[0].code,
        "git.protected_branch"
    );
}

#[test]
fn bad_policies_are_rejected_with_a_message() {
    assert!(Policy::from_yaml("preset: yolo").is_err());
    assert!(Policy::from_yaml("rules: [{action: maybe}]").is_err());
    assert!(Policy::from_yaml("unknown_key: 1").is_err());
    assert!(Policy::from_yaml("version: 2").is_err());
    assert_eq!(Policy::from_yaml("").expect("empty"), Policy::default());
}
