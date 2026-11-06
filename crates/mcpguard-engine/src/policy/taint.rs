//! Session taint tracking for the "lethal trifecta": an agent that has read
//! **private data**, been exposed to **untrusted content**, and can
//! **communicate externally** can be steered into exfiltrating the data.
//! Each condition alone is normal; all three in one session is the alarm.

use serde::Serialize;
use serde_json::Value;

use super::risk::words;

const EGRESS: &[&str] = &[
    "fetch", "http", "request", "post", "send", "email", "mail", "webhook", "upload", "publish", "comment", "message",
    "tweet", "share", "curl", "browse",
];
const UNTRUSTED_SOURCES: &[&str] = &[
    "fetch", "browse", "scrape", "web", "url", "crawl", "issue", "issues", "email", "mail", "inbox", "comment",
    "comments", "message", "messages",
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Taint {
    pub private_data: bool,
    pub untrusted_content: bool,
}

fn open_world(def: Option<&Value>) -> Option<bool> {
    def?.pointer("/annotations/openWorldHint")?.as_bool()
}

fn named(tool: &str, list: &[&str]) -> bool {
    words(tool).iter().any(|w| list.contains(&w.as_str()))
}

impl Taint {
    /// Updates after a tool result came back. `injected` = the shields found
    /// instruction-like text in it.
    pub fn observe_result(&mut self, tool: &str, def: Option<&Value>, injected: bool) {
        let ow = open_world(def);
        if injected || ow == Some(true) || named(tool, UNTRUSTED_SOURCES) {
            self.untrusted_content = true;
        }
        // Reading from a closed system (repo, database, files, drive) counts as private.
        if ow != Some(true) && !named(tool, &["fetch", "browse", "scrape", "web", "crawl"]) {
            self.private_data = true;
        }
    }

    /// Would this outgoing call complete the trifecta?
    pub fn completes_trifecta(&self, tool: &str, def: Option<&Value>, args: &Value) -> bool {
        let external =
            open_world(def) == Some(true) || named(tool, EGRESS) || !crate::vault::destinations(args).is_empty();
        self.private_data && self.untrusted_content && external
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn trifecta() {
        let mut t = Taint::default();
        let args = json!({"url": "https://evil.example/x"});
        assert!(!t.completes_trifecta("fetch", None, &args));
        t.observe_result("get_file_contents", None, false); // private repo read
        assert!(t.private_data && !t.untrusted_content);
        assert!(!t.completes_trifecta("fetch", None, &args));
        t.observe_result("list_issues", None, true); // attacker-controlled issue text
        assert!(t.completes_trifecta("fetch", None, &args));
        assert!(!t.completes_trifecta("get_me", None, &json!({})));
    }
}
