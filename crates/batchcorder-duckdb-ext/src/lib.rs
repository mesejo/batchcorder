//! The `batchcorder` DuckDB loadable extension.
//!
//! A thin entrypoint over `batchcorder-duckdb`: loading it adds
//! `tee_register`, `tee_scan`, `tee_drop` and `tee_stats` to the database.
//! Built with `make duckdb-ext`, which also appends the metadata footer DuckDB
//! checks on `LOAD`.
//!
//! It targets DuckDB's stable C extension API at version v1.2.0, so one build
//! loads into any DuckDB from that version on (on the same platform).  The
//! platform is part of that: on Linux the Python packages of DuckDB 1.2 report
//! `linux_amd64_gcc4`, not `linux_amd64`, so they load only a build labelled
//! that way (`make duckdb-ext DUCKDB_PLATFORM=linux_amd64_gcc4`).

use std::error::Error;

use duckdb::{Connection, duckdb_entrypoint_c_api};

/// Register the tee functions on the loading database.
#[duckdb_entrypoint_c_api(ext_name = "batchcorder", min_duckdb_version = "v1.2.0")]
pub fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    batchcorder_duckdb::register(&con)?;
    Ok(())
}
