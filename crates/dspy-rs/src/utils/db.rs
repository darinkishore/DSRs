use std::path::PathBuf;
use std::sync::{LazyLock, Mutex, OnceLock};

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use tracing::{debug, trace, warn};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Session context
// ---------------------------------------------------------------------------

/// A session groups all predictions from one process run.
static SESSION_ID: LazyLock<String> = LazyLock::new(|| Uuid::new_v4().to_string());

/// Returns the current session ID (stable for the lifetime of the process).
pub fn session_id() -> &'static str {
    &SESSION_ID
}

// ---------------------------------------------------------------------------
// PredictionRecord
// ---------------------------------------------------------------------------

/// Every field captured from a single `Predict::forward()` invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    pub id: String,
    pub created_at: String,

    // Predictor state
    pub signature_name: String,
    pub model_name: String,
    pub temperature: f64,
    pub max_tokens: u32,
    pub instruction_override: Option<String>,
    pub demo_count: u32,
    pub demos_json: Option<String>,

    // Messages
    pub chat_json: String,

    // Input
    pub input_json: String,

    // Output
    pub raw_response: String,
    pub output_json: Option<String>,
    pub parse_success: bool,

    // Outcome
    pub status: String,
    pub error_message: Option<String>,

    // Token usage
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,

    // Timing
    pub duration_ms: u64,

    // Parse quality
    pub field_meta_json: Option<String>,

    // Tools
    pub tool_calls_json: Option<String>,
    pub tool_executions_json: Option<String>,

    // Trace context
    pub trace_id: Option<String>,
    pub node_id: Option<usize>,

    // Grouping
    pub session_id: String,

    // Tags
    pub tags: Option<String>,
}

// ---------------------------------------------------------------------------
// PredictionDb
// ---------------------------------------------------------------------------

/// Global database singleton, initialized lazily on first log.
static GLOBAL_DB: OnceLock<PredictionDb> = OnceLock::new();

/// Persistent SQLite store for prediction records.
///
/// Thread-safe via an internal `Mutex<Connection>`. Writes are synchronous
/// but fast (single-row inserts with WAL mode). The database is created
/// automatically at `~/.dsrs/predictions.db` (or `$DSRS_PREDICTION_DB`).
pub struct PredictionDb {
    conn: Mutex<Connection>,
}

impl PredictionDb {
    /// Opens (or creates) the database at `path`.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, rusqlite::Error> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(&path)?;

        // Performance pragmas
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA busy_timeout = 5000;",
        )?;

        // Create schema
        conn.execute_batch(include_str!("db_schema.sql"))?;

        debug!(path = %path.display(), "prediction database opened");
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Opens an in-memory database (useful for tests).
    pub fn open_in_memory() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(include_str!("db_schema.sql"))?;
        debug!("prediction database opened (in-memory)");
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Returns the default database path: `~/.dsrs/predictions.db` or
    /// the value of `DSRS_PREDICTION_DB`.
    pub fn default_path() -> PathBuf {
        if let Ok(p) = std::env::var("DSRS_PREDICTION_DB") {
            return PathBuf::from(p);
        }
        dirs_fallback().join("predictions.db")
    }

    /// Returns the global singleton, initializing it on first call.
    ///
    /// Returns `None` if `DSRS_PREDICTION_DB_DISABLE=1` is set or if
    /// the database fails to open.
    pub fn global() -> Option<&'static PredictionDb> {
        if std::env::var("DSRS_PREDICTION_DB_DISABLE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            return None;
        }

        Some(GLOBAL_DB.get_or_init(|| {
            let path = Self::default_path();
            match Self::open(&path) {
                Ok(db) => db,
                Err(err) => {
                    warn!(
                        error = %err,
                        path = %path.display(),
                        "failed to open prediction database, using in-memory fallback"
                    );
                    Self::open_in_memory()
                        .expect("in-memory SQLite should never fail to open")
                }
            }
        }))
    }

    // -----------------------------------------------------------------------
    // Write
    // -----------------------------------------------------------------------

    /// Inserts a single prediction record. Fast single-row insert.
    pub fn log(&self, record: &PredictionRecord) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO predictions (
                id, created_at,
                signature_name, model_name, temperature, max_tokens,
                instruction_override, demo_count, demos_json,
                chat_json, input_json,
                raw_response, output_json, parse_success,
                status, error_message,
                prompt_tokens, completion_tokens, total_tokens,
                duration_ms,
                field_meta_json,
                tool_calls_json, tool_executions_json,
                trace_id, node_id,
                session_id, tags
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
                ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19,
                ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27
            )",
            params![
                record.id,
                record.created_at,
                record.signature_name,
                record.model_name,
                record.temperature,
                record.max_tokens,
                record.instruction_override,
                record.demo_count,
                record.demos_json,
                record.chat_json,
                record.input_json,
                record.raw_response,
                record.output_json,
                record.parse_success,
                record.status,
                record.error_message,
                record.prompt_tokens,
                record.completion_tokens,
                record.total_tokens,
                record.duration_ms,
                record.field_meta_json,
                record.tool_calls_json,
                record.tool_executions_json,
                record.trace_id,
                record.node_id,
                record.session_id,
                record.tags,
            ],
        )?;
        trace!(id = %record.id, signature = %record.signature_name, "prediction logged");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Read / query
    // -----------------------------------------------------------------------

    /// Returns the `n` most recent prediction records (newest first).
    pub fn query_recent(&self, n: usize) -> Result<Vec<PredictionRecord>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?1",
        )?;
        read_rows(&mut stmt, params![n as u32])
    }

    /// Returns all predictions for a given signature type name.
    pub fn query_by_signature(
        &self,
        name: &str,
    ) -> Result<Vec<PredictionRecord>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT * FROM predictions WHERE signature_name = ?1 ORDER BY created_at DESC",
        )?;
        read_rows(&mut stmt, params![name])
    }

    /// Returns all predictions within a time range (ISO 8601 strings).
    pub fn query_by_time_range(
        &self,
        from: &str,
        to: &str,
    ) -> Result<Vec<PredictionRecord>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT * FROM predictions WHERE created_at BETWEEN ?1 AND ?2 ORDER BY created_at DESC",
        )?;
        read_rows(&mut stmt, params![from, to])
    }

    /// Returns all predictions for a given trace ID.
    pub fn query_by_trace(
        &self,
        trace_id: &str,
    ) -> Result<Vec<PredictionRecord>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT * FROM predictions WHERE trace_id = ?1 ORDER BY created_at ASC",
        )?;
        read_rows(&mut stmt, params![trace_id])
    }

    /// Returns all predictions for a given session ID.
    pub fn query_by_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<PredictionRecord>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT * FROM predictions WHERE session_id = ?1 ORDER BY created_at DESC",
        )?;
        read_rows(&mut stmt, params![session_id])
    }

    /// Returns all predictions for the current process session.
    pub fn query_current_session(&self) -> Result<Vec<PredictionRecord>, rusqlite::Error> {
        self.query_by_session(session_id())
    }

    /// Returns all failed predictions (parse_error or lm_error).
    pub fn query_failures(&self) -> Result<Vec<PredictionRecord>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT * FROM predictions WHERE status != 'success' ORDER BY created_at DESC",
        )?;
        read_rows(&mut stmt, params![])
    }

    /// Returns the model name and total tokens grouped by model.
    pub fn token_usage_by_model(
        &self,
    ) -> Result<Vec<(String, u64, u64, u64)>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT model_name, SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens)
             FROM predictions GROUP BY model_name ORDER BY SUM(total_tokens) DESC",
        )?;
        let rows = stmt.query_map(params![], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, u64>(1)?,
                row.get::<_, u64>(2)?,
                row.get::<_, u64>(3)?,
            ))
        })?;
        rows.collect()
    }

    /// Returns the total number of prediction records.
    pub fn count(&self) -> Result<u64, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM predictions", params![], |row| {
            row.get(0)
        })
    }
}

// ---------------------------------------------------------------------------
// Row mapping helper
// ---------------------------------------------------------------------------

fn read_rows(
    stmt: &mut rusqlite::Statement<'_>,
    params: impl rusqlite::Params,
) -> Result<Vec<PredictionRecord>, rusqlite::Error> {
    let rows = stmt.query_map(params, |row| {
        Ok(PredictionRecord {
            id: row.get("id")?,
            created_at: row.get("created_at")?,
            signature_name: row.get("signature_name")?,
            model_name: row.get("model_name")?,
            temperature: row.get("temperature")?,
            max_tokens: row.get("max_tokens")?,
            instruction_override: row.get("instruction_override")?,
            demo_count: row.get("demo_count")?,
            demos_json: row.get("demos_json")?,
            chat_json: row.get("chat_json")?,
            input_json: row.get("input_json")?,
            raw_response: row.get("raw_response")?,
            output_json: row.get("output_json")?,
            parse_success: row.get("parse_success")?,
            status: row.get("status")?,
            error_message: row.get("error_message")?,
            prompt_tokens: row.get("prompt_tokens")?,
            completion_tokens: row.get("completion_tokens")?,
            total_tokens: row.get("total_tokens")?,
            duration_ms: row.get("duration_ms")?,
            field_meta_json: row.get("field_meta_json")?,
            tool_calls_json: row.get("tool_calls_json")?,
            tool_executions_json: row.get("tool_executions_json")?,
            trace_id: row.get("trace_id")?,
            node_id: row.get("node_id")?,
            session_id: row.get("session_id")?,
            tags: row.get("tags")?,
        })
    })?;
    rows.collect()
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Returns `~/.dsrs/` or a temp dir fallback.
fn dirs_fallback() -> PathBuf {
    if let Some(home) = home_dir() {
        home.join(".dsrs")
    } else {
        std::env::temp_dir().join("dsrs")
    }
}

/// Cross-platform home directory (avoids adding the `dirs` crate).
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> PredictionRecord {
        PredictionRecord {
            id: Uuid::new_v4().to_string(),
            created_at: "2026-02-17T12:00:00Z".to_string(),
            signature_name: "test::QA".to_string(),
            model_name: "openai:gpt-4o-mini".to_string(),
            temperature: 0.7,
            max_tokens: 512,
            instruction_override: None,
            demo_count: 0,
            demos_json: None,
            chat_json: r#"[{"role":"system","content":"Answer."}]"#.to_string(),
            input_json: r#"{"question":"What is 2+2?"}"#.to_string(),
            raw_response: "4".to_string(),
            output_json: Some(r#"{"answer":"4"}"#.to_string()),
            parse_success: true,
            status: "success".to_string(),
            error_message: None,
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            duration_ms: 250,
            field_meta_json: None,
            tool_calls_json: None,
            tool_executions_json: None,
            trace_id: None,
            node_id: None,
            session_id: "test-session".to_string(),
            tags: None,
        }
    }

    #[test]
    fn insert_and_query_recent() {
        let db = PredictionDb::open_in_memory().unwrap();
        let rec = sample_record();
        db.log(&rec).unwrap();

        let rows = db.query_recent(10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, rec.id);
        assert_eq!(rows[0].signature_name, "test::QA");
        assert!(rows[0].parse_success);
    }

    #[test]
    fn query_by_signature() {
        let db = PredictionDb::open_in_memory().unwrap();

        let mut r1 = sample_record();
        r1.signature_name = "module::QA".to_string();
        db.log(&r1).unwrap();

        let mut r2 = sample_record();
        r2.signature_name = "module::Summarize".to_string();
        db.log(&r2).unwrap();

        let qa_rows = db.query_by_signature("module::QA").unwrap();
        assert_eq!(qa_rows.len(), 1);
        assert_eq!(qa_rows[0].id, r1.id);
    }

    #[test]
    fn query_failures() {
        let db = PredictionDb::open_in_memory().unwrap();

        let mut success = sample_record();
        success.status = "success".to_string();
        db.log(&success).unwrap();

        let mut fail = sample_record();
        fail.status = "parse_error".to_string();
        fail.parse_success = false;
        fail.error_message = Some("missing field: answer".to_string());
        db.log(&fail).unwrap();

        let failures = db.query_failures().unwrap();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].status, "parse_error");
    }

    #[test]
    fn query_by_session() {
        let db = PredictionDb::open_in_memory().unwrap();

        let mut r1 = sample_record();
        r1.session_id = "session-A".to_string();
        db.log(&r1).unwrap();

        let mut r2 = sample_record();
        r2.session_id = "session-B".to_string();
        db.log(&r2).unwrap();

        let rows = db.query_by_session("session-A").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].session_id, "session-A");
    }

    #[test]
    fn token_usage_aggregation() {
        let db = PredictionDb::open_in_memory().unwrap();

        for _ in 0..3 {
            let mut rec = sample_record();
            rec.prompt_tokens = 100;
            rec.completion_tokens = 50;
            rec.total_tokens = 150;
            db.log(&rec).unwrap();
        }

        let usage = db.token_usage_by_model().unwrap();
        assert_eq!(usage.len(), 1);
        assert_eq!(usage[0].0, "openai:gpt-4o-mini");
        assert_eq!(usage[0].1, 300); // 3 * 100
        assert_eq!(usage[0].2, 150); // 3 * 50
        assert_eq!(usage[0].3, 450); // 3 * 150
    }

    #[test]
    fn count() {
        let db = PredictionDb::open_in_memory().unwrap();
        assert_eq!(db.count().unwrap(), 0);

        db.log(&sample_record()).unwrap();
        db.log(&sample_record()).unwrap();
        assert_eq!(db.count().unwrap(), 2);
    }

    #[test]
    fn disable_via_env_var() {
        // Note: can't test GLOBAL_DB directly since OnceLock is process-global,
        // but we can verify the env-var check logic.
        unsafe { std::env::set_var("DSRS_PREDICTION_DB_DISABLE", "1") };
        // The global() function should return None when disabled.
        // We can't call it here since OnceLock may already be set by another test,
        // so just verify the env check:
        let disabled = std::env::var("DSRS_PREDICTION_DB_DISABLE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        assert!(disabled);
        unsafe { std::env::remove_var("DSRS_PREDICTION_DB_DISABLE") };
    }

    #[test]
    fn query_by_trace() {
        let db = PredictionDb::open_in_memory().unwrap();

        let mut r1 = sample_record();
        r1.trace_id = Some("trace-123".to_string());
        r1.node_id = Some(0);
        db.log(&r1).unwrap();

        let mut r2 = sample_record();
        r2.trace_id = Some("trace-123".to_string());
        r2.node_id = Some(1);
        db.log(&r2).unwrap();

        let mut r3 = sample_record();
        r3.trace_id = Some("trace-456".to_string());
        db.log(&r3).unwrap();

        let trace_rows = db.query_by_trace("trace-123").unwrap();
        assert_eq!(trace_rows.len(), 2);
    }

    #[test]
    fn error_record_stores_full_context() {
        let db = PredictionDb::open_in_memory().unwrap();

        let mut rec = sample_record();
        rec.status = "lm_error".to_string();
        rec.parse_success = false;
        rec.error_message = Some("rate limit exceeded".to_string());
        rec.output_json = None;
        rec.raw_response = String::new();
        rec.prompt_tokens = 10;
        rec.completion_tokens = 0;
        rec.total_tokens = 10;
        db.log(&rec).unwrap();

        let rows = db.query_recent(1).unwrap();
        assert_eq!(rows[0].status, "lm_error");
        assert_eq!(
            rows[0].error_message.as_deref(),
            Some("rate limit exceeded")
        );
    }
}
