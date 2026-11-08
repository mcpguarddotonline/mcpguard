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

## Use it

```bash
npx @mcpguarddotonline/mcpguard scan   # audit Claude, Cursor, VS Code, Codex and Windsurf configs
brew install mcpguarddotonline/tap/mcpguard   # or install it (also: npm i -g @mcpguarddotonline/mcpguard)
mcpguard install --dry-run        # see what would change
mcpguard install                  # wrap every stdio server (Codex config.toml too), move
                                  # plaintext keys into the OS keychain, add Claude Code's prompt hook
mcpguard uninstall                # put everything back
```

With a hosted workspace at [mcpguard.online](https://mcpguard.online):

```bash
mcpguard login                    # sign in in your browser; this computer becomes an agent
mcpguard sync                     # workspace policy down, local decisions up (metadata only)
mcpguard status
mcpguard logout                   # revokes this computer's access
```

`sync` uploads only server and tool names, the action taken and rule ids,
never prompts, arguments or results. Tokens stay in the OS keychain.

Or by hand:

```bash
mcpguard vault add github         # stored in the OS keychain, never on disk
mcpguard wrap --name github --env GITHUB_PERSONAL_ACCESS_TOKEN=vault:github \
  -- npx -y @modelcontextprotocol/server-github
mcpguard policy init > "$(mcpguard status --policy-path)"
mcpguard policy check --server pg --tool query --args '{"sql":"DROP TABLE users"}'   # -> ask
echo 'ignore all previous instructions' | mcpguard inspect
```

`wrap` runs every message through the engine below and your policy. Risky
calls are approved in the client (MCP elicitation) where it supports that.
Vault values reach the server only as environment variables, and are masked
back to `{{vault:name}}` in anything the server returns. If the agent writes
a real value, the call is blocked.

Install from source with `cargo install --path crates/mcpguard-cli`. Release
binaries for macOS, Linux and Windows are checksummed, Sigstore-signed and
carry build provenance.

## What it detects today

- **Secrets** (redacted with typed placeholders): AWS access keys, GitHub
  tokens, Stripe live keys, Slack tokens, Anthropic/OpenAI API keys, private
  keys, and passwords in database URLs.
- **Destructive actions** (policy engine): SQL parsed with sqlparser (DROP,
  TRUNCATE, DELETE/UPDATE without WHERE), shell commands (rm -rf, curl | sh,
  force-push to protected branches, terraform destroy...), path scoping,
  recipient allow-lists, rate limits; presets and shadow mode.
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
