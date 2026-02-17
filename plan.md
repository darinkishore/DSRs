# Prediction Logging Database — Implementation Plan

## Goal

Replace the existing `ResponseCache` with a persistent SQLite database that stores **everything** from every `Predict::forward()` call. The database lives in `~/.dsrs/predictions.db` by default (configurable). On by default. Captures the full trajectory: exact messages sent to the API, predictor state (demos, instruction override), raw response, parsed output, token usage, tool calls, parse metadata — everything needed for evals, replay, optimization, and debugging.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend | SQLite (via `rusqlite`) | Embedded, zero-config, ships with the library |
| Location | `~/.dsrs/predictions.db` | Per-user, survives process restarts, configurable via env var or builder |
| Default | **On by default** | The whole point is capturing everything; users can disable if needed |
| Scope | Every `Predict::forward()` call | That's where all data converges |
| Replaces | `ResponseCache` (foyer-based) | Store everything, query what you need; cache becomes a view |

## Schema

```sql
CREATE TABLE IF NOT EXISTS predictions (
    -- Identity
    id                  TEXT PRIMARY KEY,       -- UUID v4
    created_at          TEXT NOT NULL,          -- ISO 8601 timestamp

    -- Predictor state (reconstruct the predictor)
    signature_name      TEXT NOT NULL,          -- std::any::type_name::<S>()
    model_name          TEXT NOT NULL,          -- lm.model
    instruction_override TEXT,                  -- self.instruction_override (if set)
    demo_count          INTEGER NOT NULL,       -- self.demos.len()
    demos_json          TEXT,                   -- JSON: serialized demos (input/output pairs)

    -- Messages as sent to the API (reconstruct the exact request)
    chat_json           TEXT NOT NULL,          -- JSON: full Chat (system + demos + user), via chat.to_json()

    -- Inputs
    input_json          TEXT NOT NULL,          -- JSON: serialized typed input

    -- Outputs
    raw_response        TEXT NOT NULL,          -- unprocessed LM output string
    output_json         TEXT,                   -- JSON: serialized typed output (NULL if parse failed)
    parse_success       INTEGER NOT NULL,       -- 1/0 boolean

    -- Token usage
    prompt_tokens       INTEGER NOT NULL,
    completion_tokens   INTEGER NOT NULL,
    total_tokens        INTEGER NOT NULL,

    -- Parse quality metadata
    field_meta_json     TEXT,                   -- JSON: per-field {raw_text, flags, checks}

    -- Tools
    tool_calls_json     TEXT,                   -- JSON: tool invocations
    tool_executions_json TEXT,                  -- JSON: tool execution results

    -- Trace context
    trace_id            TEXT,                   -- links to trace scope (if tracing active)
    node_id             INTEGER                 -- node ID within trace graph
);

-- Indices for common query patterns
CREATE INDEX IF NOT EXISTS idx_predictions_signature ON predictions(signature_name);
CREATE INDEX IF NOT EXISTS idx_predictions_created   ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_model     ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_trace     ON predictions(trace_id);
CREATE INDEX IF NOT EXISTS idx_predictions_success   ON predictions(parse_success);
```

## Architecture

### New module: `src/utils/db.rs`

```
PredictionDb
├── new(path: Option<PathBuf>) -> Self     // Opens/creates SQLite DB at path or ~/.dsrs/predictions.db
├── log(record: PredictionRecord) -> Result<()>  // Insert a row (non-blocking via channel)
├── query_by_signature(name) -> Vec<PredictionRecord>
├── query_by_time_range(from, to) -> Vec<PredictionRecord>
├── query_recent(n: usize) -> Vec<PredictionRecord>
├── query_by_trace(trace_id) -> Vec<PredictionRecord>
└── close()                                // Flush and close
```

### `PredictionRecord` struct

A plain data struct holding every field from the schema. Built inside `Predict::forward()` right before the return. Serialization of typed input/output happens via `serde_json::to_string()` on the `BamlValue` representations (already available through the Facet/BAML type system).

### Non-blocking writes

`PredictionDb` spawns a background writer task. `log()` sends `PredictionRecord` through a `tokio::sync::mpsc` channel. The background task batches inserts in transactions for throughput. This keeps `Predict::forward()` fast — no SQLite I/O on the hot path.

### Global access pattern

Similar to how `GLOBAL_SETTINGS` works today. A global `OnceLock<PredictionDb>` initialized at startup (or lazily on first prediction). The `Predict::forward()` call checks it and sends the record.

## Implementation Steps

### Step 1: Add dependencies
- Add `rusqlite` (with `bundled` feature) and `uuid` to `Cargo.toml`

### Step 2: Create `src/utils/db.rs`
- `PredictionRecord` struct
- `PredictionDb` with SQLite connection, schema creation, channel-based async writer
- Query methods for common access patterns (by signature, time range, trace, recent N)

### Step 3: Wire into `Predict::forward()`
- After line 325 (before the `Ok(...)`), build a `PredictionRecord` from all in-scope data
- Serialize: `chat.to_json()` for messages, `serde_json::to_value(&input)` for input, etc.
- Send to the global `PredictionDb` via the channel (fire-and-forget, non-blocking)

### Step 4: Global initialization
- Add `PredictionDb` to the global settings or as its own `OnceLock`
- Initialize with default path (`~/.dsrs/predictions.db`) unless overridden
- Add env var `DSRS_PREDICTION_DB` for path override, `DSRS_PREDICTION_DB_DISABLE=1` to turn off

### Step 5: Remove `ResponseCache` / foyer dependency
- Remove `foyer` and `tempfile` from `Cargo.toml`
- Remove `src/utils/cache.rs`
- Remove `cache` field from `LM` builder
- Remove `cache_handler` from `LM` and `DummyLM`
- Remove `inspect_history()` — replace with `PredictionDb::query_recent()`
- Update `DummyLM` to use the new DB (or skip logging for test LMs)

### Step 6: Re-export and public API
- Export `PredictionDb`, `PredictionRecord` from `lib.rs`
- Add `inspect_history`-equivalent convenience methods on `LM` that query the DB

### Step 7: Tests
- Unit tests for `PredictionDb` (insert, query, schema creation)
- Integration test: run a `Predict::forward()` call, verify the record appears in DB
- Test that disabling via env var works
- Test custom DB path

### Step 8: Update examples
- Update `07-inspect-history.rs` to use the new DB query API
- Update any other examples referencing cache
