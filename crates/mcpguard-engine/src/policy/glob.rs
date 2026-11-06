//! Tiny glob matcher: `*` (within a segment), `**` (across `/`), `?`.

pub fn matches(pattern: &str, text: &str) -> bool {
    fn go(p: &[u8], t: &[u8]) -> bool {
        // `dir/**` also matches `dir` itself.
        if p == b"/**" && t.is_empty() {
            return true;
        }
        match p.first() {
            None => t.is_empty(),
            Some(b'*') if p.get(1) == Some(&b'*') => {
                let rest = p[2..].strip_prefix(b"/").unwrap_or(&p[2..]);
                (0..=t.len()).any(|i| go(rest, &t[i..])) || (0..=t.len()).any(|i| go(&p[2..], &t[i..]))
            }
            Some(b'*') => (0..=t.len())
                .take_while(|&i| i == 0 || t[i - 1] != b'/')
                .any(|i| go(&p[1..], &t[i..])),
            Some(b'?') => !t.is_empty() && t[0] != b'/' && go(&p[1..], &t[1..]),
            Some(c) => t.first() == Some(c) && go(&p[1..], &t[1..]),
        }
    }
    go(pattern.as_bytes(), text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::matches;

    #[test]
    fn globbing() {
        assert!(matches("*", "github"));
        assert!(matches("delete_*", "delete_repo"));
        assert!(!matches("delete_*", "undelete_repo"));
        assert!(matches("release/*", "release/1.2"));
        assert!(!matches("release/*", "release/1/2"));
        assert!(matches("/home/me/projects/**", "/home/me/projects/a/b.txt"));
        assert!(matches("/home/me/projects/**", "/home/me/projects"));
        assert!(!matches("/home/me/projects/**", "/home/me/secrets/x"));
        assert!(matches("**/*.env", "a/b/.env") || matches("**/.env", "a/b/.env"));
    }
}
