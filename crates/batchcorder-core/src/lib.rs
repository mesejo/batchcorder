//! Replayable cache for read-once Arrow record-batch streams.
//!
//! [`StreamCache`] wraps an [`Upstream`] that can only be read once and lets
//! any number of independent [`StreamCacheReader`]s replay it.  Batches are
//! pulled lazily, stored in memory or spilled to an Arrow IPC file on disk
//! (see [`CacheConfig`]), and evicted once every reader has moved past them
//! as dictated by the cache's [`Bounding`].
//!
//! This crate has no Python dependency.  The `batchcorder` Python extension
//! is a thin PyO3 layer over it.

mod dataset;
mod tier;

use arrow_array::RecordBatch;
use arrow_schema::ArrowError;

pub use dataset::{Bounding, CacheConfig, DiskConfig, StreamCache, StreamCacheReader};
pub use tier::WritePolicy;

/// A source of record batches that can be read only once.
///
/// Every `Iterator<Item = Result<RecordBatch, ArrowError>> + Send`, including
/// any `RecordBatchReader + Send`, implements it.  Implement it directly when
/// pulling a batch needs extra work around the call, such as acquiring the
/// Python GIL.
///
/// `next_batch` is called while the cache's internal lock is held, so it must
/// not call back into the same [`StreamCache`].
pub trait Upstream: Send {
    /// Return the next batch, `Some(Err(_))` on failure, or `None` once the
    /// stream is exhausted.
    fn next_batch(&mut self) -> Option<Result<RecordBatch, ArrowError>>;
}

/// Wraps every blocking wait a reader makes for another reader's ingest.
///
/// The hook must call `wait` exactly once.  It runs with no cache lock held,
/// so it may release and reacquire a runtime lock around `wait`.  Python
/// bindings use it to release the GIL: a GIL-holding reader waiting on a
/// batch that a second thread is pulling from a Python upstream would
/// otherwise keep that thread from ever getting the GIL.
pub type WaitHook = fn(wait: &mut dyn FnMut());

impl<I> Upstream for I
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>> + Send,
{
    fn next_batch(&mut self) -> Option<Result<RecordBatch, ArrowError>> {
        self.next()
    }
}
