//! DuckDB table functions that *tee* one read-once Arrow stream to every
//! scan of it in a query, the DuckDB counterpart of
//! `batchcorder-datafusion`'s `TeeTableProvider`.
//!
//! The functions themselves land in a later change.  For now the crate holds
//! the probe tests (`tests/probes.rs`) that pin down how DuckDB drives a C-API
//! table function: how often it calls `bind`, `init` and the scan function per
//! query shape, and when it frees bind data.  The design rests on two of those
//! facts: every table-function reference gets its own `bind` (a
//! `NOT MATERIALIZED` CTE adds one unused bind), and the bind data of a scan
//! the optimizer prunes, or never uses, is freed before the remaining scans
//! execute.
