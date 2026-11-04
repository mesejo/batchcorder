//! Shared ingestion state, reader registration and low-water-mark eviction.
//!
//! # Lock discipline
//!
//! Readers share one `Mutex<DatasetInner>`.  No thread ever *waits* while
//! holding it: not for the upstream, not for disk I/O, not for another reader,
//! and not for the Python GIL.  Every critical section is short, pure-Rust
//! bookkeeping.  That makes it safe to block on the mutex from a thread that
//! holds the GIL (a Python `__del__`, or a host calling the Arrow C stream
//! without releasing the GIL), which in turn rules out GIL/mutex inversion.
//!
//! Three rules keep it that way:
//!
//! 1. **Ingest token.**  The upstream is taken out of `DatasetInner` by the
//!    one reader that pulls the next batch.  That reader drops the mutex,
//!    pulls (acquiring the GIL if the upstream needs it), inserts into the
//!    tier, then relocks to publish the batch and return the token.
//! 2. **Waiting readers release everything.**  A reader that needs a batch
//!    another reader is still pulling waits on a `Condvar`, wrapped in the
//!    cache's [`WaitHook`] with no lock held, so a Python host can release
//!    the GIL for the duration of the wait.
//! 3. **Evicted batches are dropped outside locks.**  See
//!    [`crate::tier::ReleasedBatches`].

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, SchemaRef};

use crate::tier::{CacheTier, ReleasedBatches, WritePolicy, other_arrow_err, poisoned_lock_err};
use crate::{Upstream, WaitHook};

// ── shared ingestion state ───────────────────────────────────────────────────

struct DatasetInner {
    cache: Arc<CacheTier>,
    /// The ingest token.  `None` while a reader is pulling from the upstream
    /// outside the lock, and for good once the upstream has panicked.
    upstream: Option<Box<dyn Upstream>>,
    ingested_count: u64,
    upstream_exhausted: bool,
    /// The upstream panicked inside `next_batch`; it was dropped and no more
    /// batches can be ingested.
    upstream_panicked: bool,
    /// Bumped whenever an ingest attempt ends or the cache is closed, so
    /// waiting readers can tell that something changed.
    ingest_epoch: u64,
    closed: bool,
    max_readers: Option<u64>,
    readers_created: u64,
    reader_positions: Vec<Weak<AtomicU64>>,
    evicted_up_to: u64,
}

impl DatasetInner {
    fn new(
        cache: Arc<CacheTier>,
        upstream: Option<Box<dyn Upstream>>,
        max_readers: Option<u64>,
    ) -> Self {
        Self {
            cache,
            upstream,
            ingested_count: 0,
            upstream_exhausted: false,
            upstream_panicked: false,
            ingest_epoch: 0,
            closed: false,
            max_readers,
            readers_created: 0,
            reader_positions: Vec::new(),
            evicted_up_to: 0,
        }
    }

    /// Register a new reader at `start_index`.  Returns the shared position
    /// tracker, or `None` when `max_readers` is unset.
    fn register_reader(
        &mut self,
        start_index: u64,
        from_start: bool,
    ) -> Result<Option<Arc<AtomicU64>>, ArrowError> {
        self.reader_positions.retain(|w| w.strong_count() > 0);
        if let Some(max) = self.max_readers
            && self.readers_created >= max
        {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Maximum number of readers ({max}) has been reached"
            )));
        }
        if from_start && self.evicted_up_to > 0 {
            return Err(ArrowError::InvalidArgumentError(
                "Cannot create a from-start reader: earliest cached batch has been evicted".into(),
            ));
        }
        if self.max_readers.is_some() {
            let pos = Arc::new(AtomicU64::new(start_index));
            self.reader_positions.push(Arc::downgrade(&pos));
            self.readers_created += 1;
            Ok(Some(pos))
        } else {
            Ok(None)
        }
    }

    /// Compute the low-water mark: the minimum position across all live
    /// readers.  Prunes dead `Weak` references as a side effect.
    fn compute_lwm(&mut self) -> Option<u64> {
        let max = self.max_readers?;
        if self.readers_created < max {
            self.reader_positions.retain(|w| w.strong_count() > 0);
            return Some(0);
        }
        let mut lwm = self.ingested_count;
        self.reader_positions.retain(|w| {
            if let Some(arc) = w.upgrade() {
                lwm = lwm.min(arc.load(Ordering::Acquire));
                true
            } else {
                false
            }
        });
        Some(lwm)
    }

    /// Evict cache entries below the low-water mark if it has advanced.
    /// The caller must drop the returned batches after unlocking.
    #[must_use = "drop the evicted batches after releasing the lock"]
    fn maybe_evict(&mut self) -> Result<ReleasedBatches, ArrowError> {
        if let Some(lwm) = self.compute_lwm()
            && lwm > self.evicted_up_to
        {
            let released = self.cache.evict_below(lwm)?;
            self.evicted_up_to = lwm;
            return Ok(released);
        }
        Ok(Vec::new())
    }
}

fn closed_err() -> ArrowError {
    ArrowError::InvalidArgumentError("Dataset has been closed".into())
}

/// Result of one pull from the upstream, made while no lock is held.
enum Pull {
    Inserted,
    Exhausted,
    Failed(ArrowError),
    Panicked,
}

/// State shared by a [`StreamCache`] and all of its readers.
struct Shared {
    state: Mutex<DatasetInner>,
    /// Notified whenever `ingest_epoch` is bumped.
    ingest_done: Condvar,
    wait_hook: Option<WaitHook>,
}

impl Shared {
    fn new(inner: DatasetInner, wait_hook: Option<WaitHook>) -> Self {
        Self {
            state: Mutex::new(inner),
            ingest_done: Condvar::new(),
            wait_hook,
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, DatasetInner>, ArrowError> {
        self.state.lock().map_err(poisoned_lock_err)
    }

    /// Block until the in-flight ingest ends (or the cache is closed).
    /// Consumes `guard`: the wait itself runs inside the wait hook with no
    /// lock held, then the caller relocks and re-checks its condition.
    fn wait_for_ingest(&self, guard: MutexGuard<'_, DatasetInner>) {
        let epoch = guard.ingest_epoch;
        drop(guard);
        let mut wait = || {
            let Ok(mut inner) = self.state.lock() else {
                return;
            };
            // Re-checked under the lock, so a notification sent between the
            // unlock above and this point is not lost.
            while inner.ingest_epoch == epoch {
                match self.ingest_done.wait(inner) {
                    Ok(next) => inner = next,
                    Err(_) => return,
                }
            }
        };
        match self.wait_hook {
            Some(hook) => hook(&mut wait),
            None => wait(),
        }
    }

    /// Make sure batch `target` is in the cache, pulling from the upstream
    /// as needed.  Returns the cache handle once it is, or `None` when the
    /// upstream ends first.
    fn ingest_up_to(&self, target: u64) -> Result<Option<Arc<CacheTier>>, ArrowError> {
        loop {
            let mut inner = self.lock()?;
            if inner.closed {
                return Err(closed_err());
            }
            if inner.ingested_count > target {
                return Ok(Some(inner.cache.clone()));
            }
            if inner.upstream_exhausted {
                return Ok(None);
            }
            if inner.upstream_panicked {
                return Err(other_arrow_err("Upstream panicked during an earlier read"));
            }
            let Some(mut upstream) = inner.upstream.take() else {
                // Another reader holds the token and is pulling right now.
                self.wait_for_ingest(inner);
                continue;
            };
            let cache = inner.cache.clone();
            drop(inner);

            // No lock held: the pull may block on I/O or the GIL, and the
            // insert may write to disk.  Only the token holder inserts, so
            // batch indices stay in upstream order.
            let pull = match catch_unwind(AssertUnwindSafe(|| upstream.next_batch())) {
                Err(_) => Pull::Panicked,
                Ok(None) => Pull::Exhausted,
                Ok(Some(Err(e))) => Pull::Failed(e),
                Ok(Some(Ok(batch))) => match cache.insert(batch) {
                    Ok(()) => Pull::Inserted,
                    Err(e) => Pull::Failed(e),
                },
            };

            // Always publish and hand the token back, even if the mutex was
            // poisoned meanwhile, so waiting readers are never stranded.
            let mut inner = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let panicked = matches!(pull, Pull::Panicked);
            if panicked {
                inner.upstream_panicked = true;
            } else {
                inner.upstream = Some(upstream);
            }
            let outcome = match pull {
                Pull::Inserted => {
                    inner.ingested_count += 1;
                    Ok(())
                }
                Pull::Exhausted => {
                    inner.upstream_exhausted = true;
                    Ok(())
                }
                Pull::Failed(e) => Err(e),
                Pull::Panicked => Err(other_arrow_err("Upstream panicked while producing a batch")),
            };
            inner.ingest_epoch += 1;
            let closed = inner.closed;
            drop(inner);
            self.ingest_done.notify_all();

            if closed {
                // `close()` ran during the pull and already cleared the
                // cache; drop anything this pull added after that.
                cache.clear();
                return Err(closed_err());
            }
            outcome?;
        }
    }
}

// ── StreamCache ────────────────────────────────────────────────────────────

/// Disk tier settings.  Batches go to a fresh subdirectory of `path`, which is
/// removed when the cache and every reader have been dropped.
#[derive(Debug, Clone)]
pub struct DiskConfig {
    pub path: PathBuf,
    /// Hard byte limit on the on-disk cache file.
    pub capacity: u64,
}

/// Construction options for [`StreamCache`].
#[derive(Debug, Clone, Default)]
pub struct CacheConfig {
    /// Byte budget of the memory tier, or of the hot layer when `disk` is set.
    /// Defaults to a fraction of system RAM.
    pub memory_capacity: Option<usize>,
    /// Spill to disk.  `None` keeps every batch in memory.
    pub disk: Option<DiskConfig>,
    /// When batches reach the disk tier.  Ignored for memory-only caches.
    pub write_policy: WritePolicy,
    /// Expected number of readers.  When set, batches every reader has moved
    /// past are evicted, and creating more readers than this is an error.
    pub max_readers: Option<u64>,
    /// Wraps every wait for another reader's ingest; see [`WaitHook`].
    pub wait_hook: Option<WaitHook>,
}

/// A cache over a read-once record-batch stream that can be replayed by many
/// independent [`StreamCacheReader`]s.
///
/// Cloning is cheap and shares the same cache.  Readers may run on different
/// threads; one of them pulls from the upstream at a time while the others
/// read already-cached batches or wait.
#[derive(Clone)]
pub struct StreamCache {
    schema: SchemaRef,
    shared: Arc<Shared>,
}

impl StreamCache {
    /// Wrap `upstream`, whose batches all have `schema`.  Nothing is pulled
    /// until a reader asks for it.  With a disk tier this performs blocking
    /// file-system I/O.
    pub fn try_new(
        upstream: Box<dyn Upstream>,
        schema: SchemaRef,
        config: CacheConfig,
    ) -> Result<Self, ArrowError> {
        if config.max_readers == Some(0) {
            return Err(ArrowError::InvalidArgumentError(
                "max_readers must be at least 1".into(),
            ));
        }
        let cache = match config.disk {
            Some(disk) => CacheTier::disk(
                &disk.path,
                disk.capacity,
                config.memory_capacity,
                config.write_policy,
            )?,
            None => CacheTier::memory(config.memory_capacity),
        };
        let inner = DatasetInner::new(Arc::new(cache), Some(upstream), config.max_readers);
        Ok(Self {
            schema,
            shared: Arc::new(Shared::new(inner, config.wait_hook)),
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Create a reader.  `from_start` replays from the first batch; otherwise
    /// the reader starts at the next batch not yet ingested.
    ///
    /// Fails when the cache is closed, when `max_readers` readers already
    /// exist, or when `from_start` is requested after eviction has begun.
    pub fn reader(&self, from_start: bool) -> Result<StreamCacheReader, ArrowError> {
        let mut inner = self.shared.lock()?;
        if inner.closed {
            return Err(closed_err());
        }
        let start_index = if from_start { 0 } else { inner.ingested_count };
        let position = inner.register_reader(start_index, from_start)?;
        Ok(StreamCacheReader {
            schema: self.schema.clone(),
            shared: self.shared.clone(),
            current_index: start_index,
            position,
        })
    }

    /// Pull every remaining upstream batch into the cache.  Returns the total
    /// number of batches ingested.
    pub fn ingest_all(&self) -> Result<u64, ArrowError> {
        self.shared.ingest_up_to(u64::MAX)?;
        Ok(self.shared.lock()?.ingested_count)
    }

    pub fn ingested_count(&self) -> Result<u64, ArrowError> {
        Ok(self.shared.lock()?.ingested_count)
    }

    pub fn upstream_exhausted(&self) -> Result<bool, ArrowError> {
        Ok(self.shared.lock()?.upstream_exhausted)
    }

    /// Release cached data and delete the disk tier.  Existing readers fail
    /// on their next call, and readers waiting for an ingest wake up and
    /// fail.  Idempotent.
    pub fn close(&self) -> Result<(), ArrowError> {
        let mut inner = self.shared.lock()?;
        if inner.closed {
            return Ok(());
        }
        inner.closed = true;
        inner.ingest_epoch += 1;
        let cache = inner.cache.clone();
        drop(inner);
        self.shared.ingest_done.notify_all();
        cache.cleanup_disk();
        cache.clear();
        Ok(())
    }
}

// ── StreamCacheReader ──────────────────────────────────────────────────────

/// An independent replay cursor over a [`StreamCache`].
///
/// Pulls from the upstream on demand, so `next()` may block on upstream I/O,
/// disk reads, another reader's ingest, or (for Python-backed upstreams) the
/// GIL.
pub struct StreamCacheReader {
    schema: SchemaRef,
    shared: Arc<Shared>,
    current_index: u64,
    /// Shared position visible to `DatasetInner` for LWM computation.
    /// `None` when `max_readers` is not set (no tracking needed).
    position: Option<Arc<AtomicU64>>,
}

impl Iterator for StreamCacheReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current_index;

        // Evict what every reader has passed.  Safe: this reader's position
        // is `idx`, so compute_lwm() returns <= idx and evict_below() never
        // removes the batch we need.
        let released = {
            let mut inner = match self.shared.state.lock() {
                Ok(g) => g,
                Err(e) => return Some(Err(other_arrow_err(format!("Mutex poisoned: {e}")))),
            };
            if inner.closed {
                return Some(Err(closed_err()));
            }
            match inner.maybe_evict() {
                Ok(released) => released,
                Err(e) => return Some(Err(e)),
            }
        };
        drop(released);

        let cache = match self.shared.ingest_up_to(idx) {
            Err(e) => return Some(Err(e)),
            Ok(None) => return None,
            Ok(Some(cache)) => cache,
        };

        // Fetch from cache; no lock held — safe for concurrent readers.
        match cache.get(idx) {
            Err(e) => Some(Err(e)),
            Ok(None) => {
                let closed = self.shared.state.lock().map(|g| g.closed).unwrap_or(false);
                if closed {
                    Some(Err(closed_err()))
                } else {
                    Some(Err(other_arrow_err(format!(
                        "Batch {idx} disappeared from the cache unexpectedly"
                    ))))
                }
            }
            Ok(Some(arc)) => {
                self.current_index += 1;
                if let Some(ref pos) = self.position {
                    pos.store(self.current_index, Ordering::Release);
                }
                Some(Ok((*arc).clone()))
            }
        }
    }
}

impl Drop for StreamCacheReader {
    fn drop(&mut self) {
        self.position.take();
        // A blocking lock is safe even with the GIL held (Python dealloc):
        // by the module's lock discipline no holder of `state` ever waits for
        // the GIL, so this wait is short and cannot form a cycle.
        let released = match self.shared.state.lock() {
            Ok(mut inner) => inner.maybe_evict().unwrap_or_default(),
            Err(_) => Vec::new(),
        };
        drop(released);
    }
}

impl arrow_array::RecordBatchReader for StreamCacheReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};

    fn make_batch(vals: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let arr = Arc::new(Int32Array::from(vals.to_vec()));
        RecordBatch::try_new(schema, vec![arr]).expect("valid batch")
    }

    fn memory_tier(capacity: usize) -> CacheTier {
        CacheTier::memory(Some(capacity))
    }

    // ── DatasetInner LWM logic ──────────────────────────────────────────────────

    fn make_dataset_inner(
        max_readers: Option<u64>,
        batch_count: u64,
    ) -> (Arc<Shared>, Arc<CacheTier>) {
        let cache = Arc::new(memory_tier(1 << 20));
        for i in 0..batch_count {
            cache.insert(make_batch(&[i as i32])).expect("insert");
        }
        let mut inner = DatasetInner::new(cache.clone(), None, max_readers);
        inner.ingested_count = batch_count;
        inner.upstream_exhausted = true;
        (Arc::new(Shared::new(inner, None)), cache)
    }

    #[test]
    fn register_reader_tracks_position() {
        let (inner_arc, _) = make_dataset_inner(Some(2), 4);
        let mut inner = inner_arc.state.lock().unwrap();
        let pos = inner.register_reader(0, true).expect("register");
        assert!(pos.is_some());
        assert_eq!(inner.readers_created, 1);
        assert_eq!(inner.reader_positions.len(), 1);
    }

    #[test]
    fn register_reader_none_max_returns_none_position() {
        let (inner_arc, _) = make_dataset_inner(None, 4);
        let mut inner = inner_arc.state.lock().unwrap();
        let pos = inner.register_reader(0, true).expect("register");
        assert!(pos.is_none());
        assert_eq!(inner.readers_created, 0);
    }

    #[test]
    fn register_reader_rejects_excess() {
        let (inner_arc, _) = make_dataset_inner(Some(1), 4);
        let mut inner = inner_arc.state.lock().unwrap();
        let _pos = inner.register_reader(0, true).expect("first ok");
        assert!(inner.register_reader(0, true).is_err());
    }

    #[test]
    fn register_reader_rejects_from_start_after_eviction() {
        let (inner_arc, _) = make_dataset_inner(Some(2), 4);
        let mut inner = inner_arc.state.lock().unwrap();
        inner.evicted_up_to = 1;
        assert!(inner.register_reader(0, true).is_err());
        assert!(inner.register_reader(1, false).is_ok());
    }

    #[test]
    fn register_reader_prunes_dead_refs() {
        let (inner_arc, _) = make_dataset_inner(Some(3), 4);
        let mut inner = inner_arc.state.lock().unwrap();
        let _pos1 = inner.register_reader(0, true).expect("r1");
        let pos2 = inner.register_reader(0, true).expect("r2");
        drop(pos2);
        let _pos3 = inner.register_reader(0, true).expect("r3");
        assert_eq!(inner.reader_positions.len(), 2);
    }

    #[test]
    fn compute_lwm_returns_none_without_max_readers() {
        let (inner_arc, _) = make_dataset_inner(None, 4);
        let mut inner = inner_arc.state.lock().unwrap();
        assert!(inner.compute_lwm().is_none());
    }

    #[test]
    fn compute_lwm_returns_zero_before_all_readers_created() {
        let (inner_arc, _) = make_dataset_inner(Some(2), 4);
        let mut inner = inner_arc.state.lock().unwrap();
        let _pos = inner.register_reader(0, true).expect("register");
        assert_eq!(inner.compute_lwm(), Some(0));
    }

    #[test]
    fn compute_lwm_returns_min_position() {
        let (inner_arc, _) = make_dataset_inner(Some(2), 4);
        let mut inner = inner_arc.state.lock().unwrap();
        let pos1 = inner.register_reader(0, true).expect("r1").unwrap();
        let pos2 = inner.register_reader(0, true).expect("r2").unwrap();
        pos1.store(3, Ordering::Release);
        pos2.store(1, Ordering::Release);
        assert_eq!(inner.compute_lwm(), Some(1));
    }

    #[test]
    fn compute_lwm_prunes_dead_readers() {
        let (inner_arc, _) = make_dataset_inner(Some(2), 4);
        let mut inner = inner_arc.state.lock().unwrap();
        let pos1 = inner.register_reader(0, true).expect("r1").unwrap();
        let pos2 = inner.register_reader(0, true).expect("r2").unwrap();
        pos1.store(3, Ordering::Release);
        drop(pos2);
        assert_eq!(inner.compute_lwm(), Some(3));
        assert_eq!(inner.reader_positions.len(), 1);
    }

    #[test]
    fn maybe_evict_advances_evicted_up_to() {
        let (inner_arc, cache) = make_dataset_inner(Some(1), 4);
        let mut inner = inner_arc.state.lock().unwrap();
        let pos = inner.register_reader(0, true).expect("register").unwrap();
        pos.store(3, Ordering::Release);
        let _ = inner.maybe_evict().expect("evict");
        assert_eq!(inner.evicted_up_to, 3);
        assert!(cache.get(0).expect("ok").is_none());
        assert!(cache.get(2).expect("ok").is_none());
        assert!(cache.get(3).expect("ok").is_some());
    }

    #[test]
    fn maybe_evict_noop_without_max_readers() {
        let (inner_arc, cache) = make_dataset_inner(None, 4);
        let mut inner = inner_arc.state.lock().unwrap();
        let _ = inner.maybe_evict().expect("evict");
        assert_eq!(inner.evicted_up_to, 0);
        assert!(cache.get(0).expect("ok").is_some());
    }

    #[test]
    fn reader_drop_triggers_eviction() {
        let (inner_arc, cache) = make_dataset_inner(Some(1), 4);
        let pos = {
            let mut inner = inner_arc.state.lock().unwrap();
            inner.register_reader(0, true).expect("register").unwrap()
        };
        pos.store(4, Ordering::Release);
        let reader = StreamCacheReader {
            schema: Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)])),
            shared: inner_arc.clone(),
            current_index: 4,
            position: Some(pos),
        };
        drop(reader);
        let inner = inner_arc.state.lock().unwrap();
        assert_eq!(inner.evicted_up_to, 4);
        assert!(cache.get(0).expect("ok").is_none());
        assert!(cache.get(3).expect("ok").is_none());
    }

    // ── StreamCache public API (native upstream, no Python) ─────────────────────

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]))
    }

    /// A plain iterator upstream over `n` single-row batches `[0], [1], ...`.
    fn native_cache(n: i32, config: CacheConfig) -> StreamCache {
        let batches: Vec<Result<RecordBatch, ArrowError>> =
            (0..n).map(|i| Ok(make_batch(&[i]))).collect();
        StreamCache::try_new(Box::new(batches.into_iter()), schema(), config).expect("cache")
    }

    fn collect(reader: StreamCacheReader) -> Vec<RecordBatch> {
        reader.collect::<Result<Vec<_>, _>>().expect("read")
    }

    fn expected(n: i32) -> Vec<RecordBatch> {
        (0..n).map(|i| make_batch(&[i])).collect()
    }

    #[test]
    fn stream_cache_replays_native_upstream() {
        let cache = native_cache(3, CacheConfig::default());
        let first = cache.reader(true).expect("r1");
        let second = cache.reader(true).expect("r2");
        assert_eq!(collect(first), expected(3));
        assert_eq!(collect(second), expected(3));
        assert!(cache.upstream_exhausted().expect("ok"));
        assert_eq!(cache.ingested_count().expect("ok"), 3);
    }

    #[test]
    fn stream_cache_readers_interleave() {
        let cache = native_cache(2, CacheConfig::default());
        let mut a = cache.reader(true).expect("a");
        let mut b = cache.reader(true).expect("b");
        assert_eq!(a.next().expect("some").expect("ok"), make_batch(&[0]));
        assert_eq!(b.next().expect("some").expect("ok"), make_batch(&[0]));
        assert_eq!(b.next().expect("some").expect("ok"), make_batch(&[1]));
        assert_eq!(a.next().expect("some").expect("ok"), make_batch(&[1]));
        assert!(a.next().is_none());
        assert!(b.next().is_none());
    }

    #[test]
    fn stream_cache_disk_tier_round_trip() {
        let dir = std::env::temp_dir().join("batchcorder_core_api_disk");
        let config = CacheConfig {
            // A 1-byte hot layer forces every read back through the disk file.
            memory_capacity: Some(1),
            disk: Some(DiskConfig {
                path: dir,
                capacity: 1 << 20,
            }),
            ..CacheConfig::default()
        };
        let cache = native_cache(4, config);
        assert_eq!(collect(cache.reader(true).expect("r1")), expected(4));
        assert_eq!(collect(cache.reader(true).expect("r2")), expected(4));
        cache.close().expect("close");
    }

    #[test]
    fn stream_cache_max_readers_evicts_and_caps() {
        let config = CacheConfig {
            max_readers: Some(2),
            ..CacheConfig::default()
        };
        let cache = native_cache(4, config);
        let a = cache.reader(true).expect("a");
        let b = cache.reader(true).expect("b");
        let err = cache.reader(true).err().expect("third reader rejected");
        assert!(matches!(err, ArrowError::InvalidArgumentError(_)));
        assert_eq!(collect(a), expected(4));
        assert_eq!(collect(b), expected(4));
        // Both readers finished and dropped, so every batch was evicted.
        let inner = cache.shared.lock().expect("lock");
        assert_eq!(inner.evicted_up_to, 4);
    }

    #[test]
    fn stream_cache_tail_reader_skips_ingested_batches() {
        let cache = native_cache(3, CacheConfig::default());
        let mut head = cache.reader(true).expect("head");
        assert_eq!(head.next().expect("some").expect("ok"), make_batch(&[0]));
        let tail = cache.reader(false).expect("tail");
        assert_eq!(collect(tail), vec![make_batch(&[1]), make_batch(&[2])]);
    }

    #[test]
    fn stream_cache_rejects_zero_max_readers() {
        let config = CacheConfig {
            max_readers: Some(0),
            ..CacheConfig::default()
        };
        let upstream: Vec<Result<RecordBatch, ArrowError>> = Vec::new();
        let err = StreamCache::try_new(Box::new(upstream.into_iter()), schema(), config)
            .err()
            .expect("rejected");
        assert!(matches!(err, ArrowError::InvalidArgumentError(_)));
    }

    #[test]
    fn stream_cache_close_fails_readers() {
        let cache = native_cache(2, CacheConfig::default());
        let mut reader = cache.reader(true).expect("reader");
        cache.close().expect("close");
        cache.close().expect("idempotent");
        assert!(matches!(
            reader.next(),
            Some(Err(ArrowError::InvalidArgumentError(_)))
        ));
        assert!(cache.reader(true).is_err());
        assert!(cache.ingest_all().is_err());
    }

    #[test]
    fn stream_cache_upstream_error_propagates() {
        let upstream: Vec<Result<RecordBatch, ArrowError>> = vec![
            Ok(make_batch(&[0])),
            Err(ArrowError::ComputeError("boom".into())),
        ];
        let cache = StreamCache::try_new(
            Box::new(upstream.into_iter()),
            schema(),
            CacheConfig::default(),
        )
        .expect("cache");
        let mut reader = cache.reader(true).expect("reader");
        assert!(reader.next().expect("some").is_ok());
        assert!(matches!(
            reader.next(),
            Some(Err(ArrowError::ComputeError(_)))
        ));
    }

    #[test]
    fn stream_cache_reader_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<StreamCache>();
        assert_send::<StreamCacheReader>();
    }

    // ── concurrent ingestion (ingest token + condvar) ───────────────────────────

    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc::{Receiver, Sender, channel};
    use std::thread;
    use std::time::Duration;

    /// Fails the test instead of hanging when a deadlock regresses.
    const TIMEOUT: Duration = Duration::from_secs(10);

    enum Item {
        Batch(i32),
        Error,
        Panic,
    }

    /// An upstream that signals `entered` when a pull starts and then blocks
    /// until the test sends on `gate`, so a test can hold a pull in flight.
    struct GatedUpstream {
        items: std::vec::IntoIter<Item>,
        entered: Sender<()>,
        gate: Receiver<()>,
        pulls: Arc<AtomicUsize>,
    }

    impl Upstream for GatedUpstream {
        fn next_batch(&mut self) -> Option<Result<RecordBatch, ArrowError>> {
            self.pulls.fetch_add(1, Ordering::SeqCst);
            let _ = self.entered.send(());
            let _ = self.gate.recv_timeout(TIMEOUT);
            match self.items.next()? {
                Item::Batch(v) => Some(Ok(make_batch(&[v]))),
                Item::Error => Some(Err(ArrowError::ComputeError("boom".into()))),
                Item::Panic => panic!("upstream panicked on purpose"),
            }
        }
    }

    struct Gated {
        cache: StreamCache,
        entered: Receiver<()>,
        gate: Sender<()>,
        pulls: Arc<AtomicUsize>,
    }

    fn gated_cache(items: Vec<Item>, config: CacheConfig) -> Gated {
        let (entered_tx, entered) = channel();
        let (gate, gate_rx) = channel();
        let pulls = Arc::new(AtomicUsize::new(0));
        let upstream = GatedUpstream {
            items: items.into_iter(),
            entered: entered_tx,
            gate: gate_rx,
            pulls: pulls.clone(),
        };
        let cache = StreamCache::try_new(Box::new(upstream), schema(), config).expect("cache");
        Gated {
            cache,
            entered,
            gate,
            pulls,
        }
    }

    /// Run `f` on a new thread and return a receiver for its result.
    fn spawn<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Receiver<T> {
        let (tx, rx) = channel();
        thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx
    }

    fn next_value(reader: &mut StreamCacheReader) -> Option<Result<RecordBatch, ArrowError>> {
        reader.next()
    }

    #[test]
    fn cached_batches_readable_while_pull_in_flight() {
        let g = gated_cache(vec![Item::Batch(0), Item::Batch(1)], CacheConfig::default());
        let mut ahead = g.cache.reader(true).expect("ahead");
        let mut behind = g.cache.reader(true).expect("behind");

        // Batch 0 is pulled and cached.
        g.gate.send(()).expect("gate");
        assert_eq!(ahead.next().expect("some").expect("ok"), make_batch(&[0]));
        g.entered.recv_timeout(TIMEOUT).expect("first pull entered");

        // `ahead` starts pulling batch 1 and blocks inside the upstream.
        let ahead_rx = spawn(move || next_value(&mut ahead).map(|r| r.map_err(|e| e.to_string())));
        g.entered
            .recv_timeout(TIMEOUT)
            .expect("second pull entered");

        // The mutex is not held during the pull, so the cached batch is
        // served immediately, as are the cache's own getters.
        let got = spawn(move || behind.next().map(|r| r.map_err(|e| e.to_string())))
            .recv_timeout(TIMEOUT)
            .expect("cached read must not wait for the in-flight pull");
        assert_eq!(got, Some(Ok(make_batch(&[0]))));
        assert_eq!(g.cache.ingested_count().expect("ok"), 1);

        g.gate.send(()).expect("gate");
        let got = ahead_rx.recv_timeout(TIMEOUT).expect("pull completes");
        assert_eq!(got, Some(Ok(make_batch(&[1]))));
    }

    #[test]
    fn waiter_wakes_and_shares_the_pulled_batch() {
        let g = gated_cache(vec![Item::Batch(7)], CacheConfig::default());
        let mut a = g.cache.reader(true).expect("a");
        let mut b = g.cache.reader(true).expect("b");

        let a_rx = spawn(move || next_value(&mut a).map(|r| r.map_err(|e| e.to_string())));
        g.entered.recv_timeout(TIMEOUT).expect("a is pulling");
        // `b` needs the same batch: it waits on the condvar, it does not pull.
        let b_rx = spawn(move || next_value(&mut b).map(|r| r.map_err(|e| e.to_string())));
        thread::sleep(Duration::from_millis(50));
        g.gate.send(()).expect("gate");

        assert_eq!(
            a_rx.recv_timeout(TIMEOUT).expect("a"),
            Some(Ok(make_batch(&[7])))
        );
        assert_eq!(
            b_rx.recv_timeout(TIMEOUT).expect("b"),
            Some(Ok(make_batch(&[7])))
        );
        assert_eq!(g.pulls.load(Ordering::SeqCst), 1, "batch pulled once");
    }

    #[test]
    fn upstream_error_wakes_waiter_which_then_pulls() {
        let g = gated_cache(vec![Item::Error, Item::Batch(1)], CacheConfig::default());
        let mut a = g.cache.reader(true).expect("a");
        let mut b = g.cache.reader(true).expect("b");

        let a_rx = spawn(move || next_value(&mut a).map(|r| r.map_err(|e| e.to_string())));
        g.entered.recv_timeout(TIMEOUT).expect("a is pulling");
        let b_rx = spawn(move || next_value(&mut b).map(|r| r.map_err(|e| e.to_string())));
        thread::sleep(Duration::from_millis(50));
        g.gate.send(()).expect("gate for a");

        // `a` gets the error; the token goes back, so `b` wakes and pulls.
        let a_got = a_rx.recv_timeout(TIMEOUT).expect("a");
        assert!(matches!(a_got, Some(Err(ref e)) if e.contains("boom")));
        g.entered.recv_timeout(TIMEOUT).expect("b is pulling");
        g.gate.send(()).expect("gate for b");
        assert_eq!(
            b_rx.recv_timeout(TIMEOUT).expect("b"),
            Some(Ok(make_batch(&[1])))
        );
    }

    #[test]
    fn close_during_pull_wakes_waiters_and_fails_puller() {
        let g = gated_cache(vec![Item::Batch(0)], CacheConfig::default());
        let mut a = g.cache.reader(true).expect("a");
        let mut b = g.cache.reader(true).expect("b");

        let a_rx = spawn(move || next_value(&mut a).map(|r| r.map_err(|e| e.to_string())));
        g.entered.recv_timeout(TIMEOUT).expect("a is pulling");
        let b_rx = spawn(move || next_value(&mut b).map(|r| r.map_err(|e| e.to_string())));
        thread::sleep(Duration::from_millis(50));

        g.cache.close().expect("close");
        let b_got = b_rx.recv_timeout(TIMEOUT).expect("waiter wakes on close");
        assert!(matches!(b_got, Some(Err(ref e)) if e.contains("closed")));

        g.gate.send(()).expect("gate");
        let a_got = a_rx.recv_timeout(TIMEOUT).expect("puller returns");
        assert!(matches!(a_got, Some(Err(ref e)) if e.contains("closed")));
        assert_eq!(g.cache.ingested_count().expect("ok"), 1);
    }

    #[test]
    fn upstream_panic_becomes_error_and_is_sticky() {
        let g = gated_cache(vec![Item::Panic, Item::Batch(1)], CacheConfig::default());
        let mut a = g.cache.reader(true).expect("a");
        g.gate.send(()).expect("gate");
        let err = a
            .next()
            .expect("some")
            .expect_err("panic surfaces as error");
        assert!(err.to_string().contains("panicked"));
        // The upstream is gone; later reads fail instead of hanging.
        let mut b = g.cache.reader(true).expect("b");
        let err = b.next().expect("some").expect_err("still failing");
        assert!(err.to_string().contains("panicked"));
        assert_eq!(g.pulls.load(Ordering::SeqCst), 1);
    }

    static HOOK_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn counting_hook(wait: &mut dyn FnMut()) {
        HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
        wait();
    }

    #[test]
    fn wait_hook_wraps_condvar_waits() {
        let config = CacheConfig {
            wait_hook: Some(counting_hook),
            ..CacheConfig::default()
        };
        let g = gated_cache(vec![Item::Batch(3)], config);
        let mut a = g.cache.reader(true).expect("a");
        let mut b = g.cache.reader(true).expect("b");

        let a_rx = spawn(move || next_value(&mut a).map(|r| r.map_err(|e| e.to_string())));
        g.entered.recv_timeout(TIMEOUT).expect("a is pulling");
        let before = HOOK_CALLS.load(Ordering::SeqCst);
        let b_rx = spawn(move || next_value(&mut b).map(|r| r.map_err(|e| e.to_string())));
        // Wait until `b` is parked inside the hook.
        for _ in 0..1000 {
            if HOOK_CALLS.load(Ordering::SeqCst) > before {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            HOOK_CALLS.load(Ordering::SeqCst) > before,
            "b waited via the hook"
        );
        g.gate.send(()).expect("gate");
        assert_eq!(
            a_rx.recv_timeout(TIMEOUT).expect("a"),
            Some(Ok(make_batch(&[3])))
        );
        assert_eq!(
            b_rx.recv_timeout(TIMEOUT).expect("b"),
            Some(Ok(make_batch(&[3])))
        );
    }

    /// An ungated upstream that counts pulls and yields to other threads.
    struct SlowUpstream {
        next: i32,
        total: i32,
        pulls: Arc<AtomicUsize>,
    }

    impl Upstream for SlowUpstream {
        fn next_batch(&mut self) -> Option<Result<RecordBatch, ArrowError>> {
            self.pulls.fetch_add(1, Ordering::SeqCst);
            thread::sleep(Duration::from_micros(200));
            if self.next == self.total {
                return None;
            }
            self.next += 1;
            Some(Ok(make_batch(&[self.next - 1])))
        }
    }

    #[test]
    fn concurrent_readers_see_every_batch_in_order_once() {
        const N: i32 = 200;
        const READERS: u64 = 8;
        let pulls = Arc::new(AtomicUsize::new(0));
        let upstream = SlowUpstream {
            next: 0,
            total: N,
            pulls: pulls.clone(),
        };
        let config = CacheConfig {
            max_readers: Some(READERS),
            ..CacheConfig::default()
        };
        let cache = StreamCache::try_new(Box::new(upstream), schema(), config).expect("cache");
        let receivers: Vec<_> = (0..READERS)
            .map(|_| {
                let reader = cache.reader(true).expect("reader");
                spawn(move || collect(reader))
            })
            .collect();
        for rx in receivers {
            assert_eq!(rx.recv_timeout(TIMEOUT * 3).expect("reader"), expected(N));
        }
        // N batches plus the one pull that saw the end of the stream.
        assert_eq!(pulls.load(Ordering::SeqCst), N as usize + 1);
        assert_eq!(cache.ingested_count().expect("ok"), N as u64);
        // Every reader finished, so everything was evicted.
        assert_eq!(cache.shared.lock().expect("lock").evicted_up_to, N as u64);
    }
}
