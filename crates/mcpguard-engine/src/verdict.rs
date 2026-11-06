use serde::Serialize;

/// What the caller should do with an inspected message.
///
/// Variants are ordered by severity (`Allow < … < Block`); when several rules
/// match, the strictest action wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    /// Pass through unchanged.
    #[default]
    Allow,
    /// Pass through, but mark the content as untrusted data for the model.
    Annotate,
    /// Pass through with sensitive spans replaced by placeholders.
    Redact,
    /// Hold until a human approves.
    Ask,
    /// Refuse.
    Block,
}

/// Which way a message is travelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Tool-call arguments leaving the agent for an MCP server.
    ToServer,
    /// Tool results and tool definitions entering the model's context.
    ToClient,
}

/// Category of a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Secret,
    Injection,
    Input,
}

/// One rule match.
///
/// `start`/`end` are byte offsets into the **normalized** text (see
/// [`Verdict::output`]), except for `injection.hidden_unicode`, whose offsets
/// point into the original input (the hidden characters are gone after
/// normalization).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Finding {
    pub rule_id: &'static str,
    pub kind: Kind,
    pub action: Action,
    pub start: usize,
    pub end: usize,
    /// Placeholder label for secrets, e.g. `github_token`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<&'static str>,
    pub note: &'static str,
}

/// The engine's decision for one message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Verdict {
    pub action: Action,
    pub findings: Vec<Finding>,
    /// Text to forward: secrets replaced when `action` is `Redact` or
    /// stricter, otherwise the normalized input (invisible characters removed).
    #[serde(skip)]
    pub output: String,
}
