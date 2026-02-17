//! LM response caching and prediction logging.
//!
//! The [`ResponseCache`] provides a hybrid memory + disk cache backed by
//! [foyer](https://docs.rs/foyer). It also maintains a sliding window of recent
//! entries for [`LM::inspect_history`](crate::LM::inspect_history).
//!
//! The [`PredictionDb`] is a persistent SQLite store that captures every
//! `Predict::forward()` call — inputs, outputs, errors, token usage, timing,
//! and trace context. On by default at `~/.dsrs/predictions.db`.

pub mod cache;
pub mod db;
pub mod serde_utils;
pub mod telemetry;

pub use cache::{Cache, CacheEntry, ResponseCache};
pub use db::{PredictionDb, PredictionRecord, session_id};
pub use serde_utils::get_iter_from_value;
pub use telemetry::{TelemetryInitError, init_tracing, truncate};
