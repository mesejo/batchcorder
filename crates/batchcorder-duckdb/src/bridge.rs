//! Move Arrow data from the workspace's arrow (59) to the arrow duckdb-rs
//! links (60), through the Arrow C Data Interface.
//!
//! No data is copied: each array is exported as an `FFI_ArrowArray` /
//! `FFI_ArrowSchema` pair and imported on the other side, which takes over
//! the buffers and the `release` callback.  Remove this module once the
//! workspace moves to the arrow major duckdb-rs uses.

use std::error::Error;
use std::mem::transmute;
use std::sync::Arc;

use arrow_array::ffi::{FFI_ArrowArray as Array59, FFI_ArrowSchema as Schema59, to_ffi};
use arrow_array::{Array as _, RecordBatch as Batch59, StructArray as Struct59};
use arrow_schema::Schema as ArrowSchema59;
use duckdb::arrow::array::{RecordBatch, StructArray};
use duckdb::arrow::datatypes::{Schema, SchemaRef};
use duckdb::arrow::ffi::{FFI_ArrowArray as Array60, FFI_ArrowSchema as Schema60, from_ffi};

pub(crate) type BoxError = Box<dyn Error>;

/// Re-type a C Data Interface struct from arrow 59 to arrow 60.
///
/// SAFETY (for every caller): both types are the `#[repr(C)]` structs the
/// Arrow C Data Interface specifies, so their layouts are identical;
/// `transmute` checks the sizes at compile time.  Ownership, including the
/// `release` callback, moves with the value, so it is released exactly once,
/// by arrow 60.
fn schema_59_to_60(schema: Schema59) -> Schema60 {
    unsafe { transmute::<Schema59, Schema60>(schema) }
}

fn array_59_to_60(array: Array59) -> Array60 {
    // SAFETY: see `schema_59_to_60`.
    unsafe { transmute::<Array59, Array60>(array) }
}

/// The arrow-60 form of an arrow-59 schema.
pub(crate) fn schema(schema: &ArrowSchema59) -> Result<SchemaRef, BoxError> {
    let ffi = schema_59_to_60(Schema59::try_from(schema)?);
    Ok(Arc::new(Schema::try_from(&ffi)?))
}

/// The arrow-60 form of an arrow-59 batch, sharing its buffers.
pub(crate) fn batch(batch: &Batch59) -> Result<RecordBatch, BoxError> {
    let (array, schema) = to_ffi(&Struct59::from(batch.clone()).to_data())?;
    let (array, schema) = (array_59_to_60(array), schema_59_to_60(schema));
    // SAFETY: `array` and `schema` were just exported together by `to_ffi`
    // and describe the same data.
    let data = unsafe { from_ffi(array, &schema)? };
    Ok(RecordBatch::from(StructArray::from(data)))
}
