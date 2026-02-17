CREATE TABLE IF NOT EXISTS predictions (
    -- Identity
    id                  TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,

    -- Predictor state
    signature_name      TEXT NOT NULL,
    model_name          TEXT NOT NULL,
    temperature         REAL NOT NULL,
    max_tokens          INTEGER NOT NULL,
    instruction_override TEXT,
    demo_count          INTEGER NOT NULL,
    demos_json          TEXT,

    -- Messages as sent to the API
    chat_json           TEXT NOT NULL,

    -- Input
    input_json          TEXT NOT NULL,

    -- Output
    raw_response        TEXT NOT NULL,
    output_json         TEXT,
    parse_success       INTEGER NOT NULL,

    -- Outcome
    status              TEXT NOT NULL,
    error_message       TEXT,

    -- Token usage
    prompt_tokens       INTEGER NOT NULL,
    completion_tokens   INTEGER NOT NULL,
    total_tokens        INTEGER NOT NULL,

    -- Timing
    duration_ms         INTEGER NOT NULL,

    -- Parse quality
    field_meta_json     TEXT,

    -- Tools
    tool_calls_json     TEXT,
    tool_executions_json TEXT,

    -- Trace context
    trace_id            TEXT,
    node_id             INTEGER,

    -- Grouping
    session_id          TEXT NOT NULL,

    -- User metadata
    tags                TEXT
);

-- Indices for common query patterns
CREATE INDEX IF NOT EXISTS idx_predictions_signature  ON predictions(signature_name);
CREATE INDEX IF NOT EXISTS idx_predictions_created    ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_model      ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_trace      ON predictions(trace_id);
CREATE INDEX IF NOT EXISTS idx_predictions_success    ON predictions(parse_success);
CREATE INDEX IF NOT EXISTS idx_predictions_session    ON predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_status     ON predictions(status);
