//! SQL argument analysis: finds destructive statements by parsing, not by
//! keyword search (so `SELECT 'drop table'` is fine and `dRoP/**/TABLE` is not).

use sqlparser::ast::{ObjectType, Statement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

/// What a SQL string does, most dangerous first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SqlFinding {
    DropDatabase,
    Drop(String),
    Truncate,
    DeleteWithoutWhere,
    UpdateWithoutWhere,
    AlterDrop,
    Grant,
    /// Didn't parse. Treated as unknown, never as safe.
    Unparsed,
}

impl SqlFinding {
    pub fn code(&self) -> &'static str {
        match self {
            SqlFinding::DropDatabase => "sql.drop_database",
            SqlFinding::Drop(_) => "sql.drop",
            SqlFinding::Truncate => "sql.truncate",
            SqlFinding::DeleteWithoutWhere => "sql.delete_without_where",
            SqlFinding::UpdateWithoutWhere => "sql.update_without_where",
            SqlFinding::AlterDrop => "sql.alter_drop",
            SqlFinding::Grant => "sql.grant",
            SqlFinding::Unparsed => "sql.unparsed",
        }
    }

    /// Short words used in policy files (`sql: [drop, truncate]`).
    pub fn keyword(&self) -> &'static str {
        match self {
            SqlFinding::DropDatabase | SqlFinding::Drop(_) => "drop",
            SqlFinding::Truncate => "truncate",
            SqlFinding::DeleteWithoutWhere => "delete_all",
            SqlFinding::UpdateWithoutWhere => "update_all",
            SqlFinding::AlterDrop => "alter_drop",
            SqlFinding::Grant => "grant",
            SqlFinding::Unparsed => "unparsed",
        }
    }

    pub fn catastrophic(&self) -> bool {
        matches!(
            self,
            SqlFinding::DropDatabase | SqlFinding::Drop(_) | SqlFinding::Truncate
        )
    }
}

/// Heuristic: does this look like SQL at all? (Avoids parsing prose.)
pub fn looks_like_sql(s: &str) -> bool {
    let first = s
        .trim_start()
        .split(|c: char| !c.is_ascii_alphabetic())
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    matches!(
        first.as_str(),
        "SELECT"
            | "INSERT"
            | "UPDATE"
            | "DELETE"
            | "DROP"
            | "TRUNCATE"
            | "ALTER"
            | "CREATE"
            | "GRANT"
            | "REVOKE"
            | "WITH"
            | "MERGE"
            | "REPLACE"
            | "UPSERT"
    )
}

pub fn analyze(sql: &str) -> Vec<SqlFinding> {
    let Ok(stmts) = Parser::parse_sql(&GenericDialect {}, sql) else {
        return vec![SqlFinding::Unparsed];
    };
    let mut out = Vec::new();
    for st in stmts {
        match st {
            Statement::Drop { object_type, .. } => {
                if matches!(object_type, ObjectType::Database | ObjectType::Schema) {
                    out.push(SqlFinding::DropDatabase);
                } else {
                    out.push(SqlFinding::Drop(format!("{object_type:?}").to_ascii_lowercase()));
                }
            }
            Statement::Truncate(_) => out.push(SqlFinding::Truncate),
            Statement::Delete(d) if d.selection.is_none() => out.push(SqlFinding::DeleteWithoutWhere),
            Statement::Update(u) if u.selection.is_none() => out.push(SqlFinding::UpdateWithoutWhere),
            Statement::AlterTable(a) if a.operations.iter().any(|op| format!("{op:?}").starts_with("Drop")) => {
                out.push(SqlFinding::AlterDrop)
            }
            Statement::Grant(_) | Statement::Revoke(_) => out.push(SqlFinding::Grant),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_destructive_statements() {
        assert_eq!(analyze("DROP TABLE users"), [SqlFinding::Drop("table".into())]);
        assert_eq!(analyze("drop database prod;"), [SqlFinding::DropDatabase]);
        assert_eq!(analyze("TRUNCATE orders"), [SqlFinding::Truncate]);
        assert_eq!(analyze("DELETE FROM users"), [SqlFinding::DeleteWithoutWhere]);
        assert_eq!(
            analyze("UPDATE users SET admin = true"),
            [SqlFinding::UpdateWithoutWhere]
        );
        assert_eq!(analyze("ALTER TABLE t DROP COLUMN c"), [SqlFinding::AlterDrop]);
        assert_eq!(analyze("SELECT 1; DROP TABLE x"), [SqlFinding::Drop("table".into())]);
    }

    #[test]
    fn safe_statements_and_strings() {
        assert!(analyze("SELECT 'drop table users' AS s").is_empty());
        assert!(analyze("DELETE FROM users WHERE id = 4").is_empty());
        assert!(analyze("UPDATE t SET a = 1 WHERE id = 2").is_empty());
        assert!(looks_like_sql("  select * from t"));
        assert!(!looks_like_sql("Please drop by tomorrow"));
    }
}
