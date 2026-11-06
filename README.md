# MCP-Guard

**A firewall between your AI agent and its tools.** MCP-Guard inspects
[Model Context Protocol](https://modelcontextprotocol.io) traffic and stops
prompt injection, secret leaks and destructive actions before they happen.
It works with Claude, Cursor, Codex, GitHub Copilot and any MCP client.

This repository is the open-source core (Apache-2.0):

| Crate | What it is |
|---|---|
| [`mcpguard-engine`](crates/mcpguard-engine) | The detection engine: normalization, secret rules, prompt-injection signatures, verdicts |
| [`mcpguard-cli`](crates/mcpguard-cli) | The `mcpguard` command-line tool. It runs entirely on your machine; nothing is sent anywhere |

The hosted gateway, dashboard and phone approvals are at
[mcpguard.online](https://mcpguard.online).

## Status

Early development. Available now:

```bash
cargo install --path crates/mcpguard-cli
echo 'ignore all previous instructions' | mcpguard inspect
mcpguard inspect --direction to-server --output --text 'postgres://app:hunter22@db/prod'
# -> postgres://app:«SECRET:db_password#1»@db/prod
```

Coming next: `mcpguard scan` (audit your MCP configs), `mcpguard wrap`
(guard a stdio MCP server) and `mcpguard install`.

## What it detects today

- **Secrets** (redacted with typed placeholders): AWS access keys, GitHub
  tokens, Stripe live keys, Slack tokens, Anthropic/OpenAI API keys, private
  keys, and passwords in database URLs.
- **Prompt injection** in content entering the model: instruction overrides,
  fake role markers, identity resets, tool coercion, markdown-image
  exfiltration, and instructions hidden in invisible Unicode.

## Development

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Found a bypass? Add a failing test and open a PR. Security issues: see [SECURITY.md](SECURITY.md).

## License

[Apache-2.0](LICENSE)
