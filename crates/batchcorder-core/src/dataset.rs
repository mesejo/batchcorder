//! Shared ingestion state, reader registration and low-water-mark eviction.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, SchemaRef};

use crate::Upstream;
use crate::tier::{CacheTier, WritePolicy, other_arrow_err, poisoned_lock_err};

// ── shared ingestion state ───────────────────────────────────────────────────

struct DatasetInner {
    cache: Arc<CacheTier>,
    upstream: Option<Box<dyn Upstream>>,
    ingested_count: u64,
    upstream_exhausted: bool,
    closed: bool,
    max_readers: Option<u64>,
    readers_created: u64,
    reader_positions: Vec<Weak<AtomicU64>>,
    evicted_up_to: u64,
}

impl DatasetInner {
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
    fn maybe_evict(&mut self) -> Result<(), ArrowError> {
        if let Some(lwm) = self.compute_lwm()
            && lwm > self.evicted_up_to
        {
            self.cache.evict_below(lwm)?;
            self.evicted_up_to = lwm;
        }
        Ok(())
    }

    fn ingest_up_to(&mut self, target_index: u64) -> Result<bool, ArrowError> {
        if self.closed {
            return Err(ArrowError::InvalidArgumentError(
                "Dataset has been closed".into(),
            ));
        }
        while self.ingested_count <= target_index {
            if self.upstream_exhausted {
                return Ok(false);
            }
            // Python-backed upstreams acquire the GIL inside `next_batch`.
            let next_result = self.upstream.as_mut().and_then(|u| u.next_batch());
            let batch = match next_result {
                None => {
                    self.upstream_exhausted = true;
                    return Ok(false);
                }
                Some(Err(e)) => return Err(e),
                Some(Ok(b)) => b,
            };
            self.cache.insert(batch)?;
            self.ingested_count += 1;
        }
        Ok(true)
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
}

/// A cache over a read-once record-batch stream that can be replayed by many
/// independent [`StreamCacheReader`]s.
///
/// Cloning is cheap and shares the same cache.
#[derive(Clone)]
pub struct StreamCache {
    schema: SchemaRef,
    inner: Arc<Mutex<DatasetInner>>,
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
        let inner = DatasetInner {
            cache: Arc::new(cache),
            upstream: Some(upstream),
            ingested_count: 0,
            upstream_exhausted: false,
            closed: false,
            max_readers: config.max_readers,
            readers_created: 0,
            reader_positions: Vec::new(),
            evicted_up_to: 0,
        };
        Ok(Self {
            schema,
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn lock(&self) -> Result<MutexGuard<'_, DatasetInner>, ArrowError> {
        self.inner.lock().map_err(poisoned_lock_err)
    }

    /// Create a reader.  `from_start` replays from the first batch; otherwise
    /// the reader starts at the next batch not yet ingested.
    ///
    /// Fails when the cache is closed, when `max_readers` readers already
    /// exist, or when `from_start` is requested after eviction has begun.
    pub fn reader(&self, from_start: bool) -> Result<StreamCacheReader, ArrowError> {
        let mut inner = self.lock()?;
        if inner.closed {
            return Err(ArrowError::InvalidArgumentError(
                "Dataset has been closed".into(),
            ));
        }
        let start_index = if from_start { 0 } else { inner.ingested_count };
        let position = inner.register_reader(start_index, from_start)?;
        Ok(StreamCacheReader {
            schema: self.schema.clone(),
            inner: self.inner.clone(),
            current_index: start_index,
            position,
        })
    }

    /// Pull every remaining upstream batch into the cache.  Returns the total
    /// number of batches ingested.
    pub fn ingest_all(&self) -> Result<u64, ArrowError> {
        let mut inner = self.lock()?;
        inner.ingest_up_to(u64::MAX)?;
        Ok(inner.ingested_count)
    }

    pub fn ingested_count(&self) -> Result<u64, ArrowError> {
        Ok(self.lock()?.ingested_count)
    }

    pub fn upstream_exhausted(&self) -> Result<bool, ArrowError> {
        Ok(self.lock()?.upstream_exhausted)
    }

    /// Release cached data and delete the disk tier.  Existing readers fail
    /// on their next call.  Idempotent.
    pub fn close(&self) -> Result<(), ArrowError> {
        let mut inner = self.lock()?;
        if inner.closed {
            return Ok(());
        }
        inner.closed = true;
        let cache = inner.cache.clone();
        drop(inner);
        cache.cleanup_disk();
        cache.clear();
        Ok(())
    }
}

// ── StreamCacheReader ──────────────────────────────────────────────────────

/// An independent replay cursor over a [`StreamCache`].
///
/// Pulls from the upstream on demand, so `next()` may block on upstream I/O,
/// disk reads, or (for Python-backed upstreams) the GIL.
pub struct StreamCacheReader {
    schema: SchemaRef,
    inner: Arc<Mutex<DatasetInner>>,
    current_index: u64,
    /// Shared position visible to `DatasetInner` for LWM computation.
    /// `None` when `max_readers` is not set (no tracking needed).
    position: Option<Arc<AtomicU64>>,
}

impl Iterator for StreamCacheReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current_index;

        // Ensure the batch has been ingested and clone the cache handle.
        let cache = {
            let mut inner = match self.inner.lock() {
                Ok(g) => g,
                Err(e) => return Some(Err(other_arrow_err(format!("Mutex poisoned: {e}")))),
            };
            if inner.closed {
                return Some(Err(ArrowError::InvalidArgumentError(
                    "Dataset has been closed".into(),
                )));
            }
            // Safe: this reader's position is `idx`, so compute_lwm() returns
            // <= idx and evict_below() never removes the batch we need.
            if let Err(e) = inner.maybe_evict() {
                return Some(Err(e));
            }
            match inner.ingest_up_to(idx) {
                Err(e) => return Some(Err(e)),
                Ok(false) => return None,
                Ok(true) => {}
            }
            inner.cache.clone()
        }; // mutex released before I/O

        // Fetch from cache; no lock held — safe for concurrent readers.
        match cache.get(idx) {
            Err(e) => Some(Err(e)),
            Ok(None) => {
                let closed = self.inner.lock().map(|g| g.closed).unwrap_or(false);
                if closed {
                    Some(Err(ArrowError::InvalidArgumentError(
                        "Dataset has been closed".into(),
                    )))
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
        // try_lock, never lock: Drop runs with the GIL held (Python dealloc),
        // while another thread may hold `inner` inside `ingest_up_to` waiting
        // to reacquire the GIL for `upstream.next()` — a blocking lock here
        // would deadlock.  Eviction is opportunistic; if the lock is busy the
        // next `next()` call on any surviving reader evicts instead.
        if let Ok(mut inner) = self.inner.try_lock() {
            let _ = inner.maybe_evict();
        }
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
    ) -> (Arc<Mutex<DatasetInner>>, Arc<CacheTier>) {
        let cache = Arc::new(memory_tier(1 << 20));
        for i in 0..batch_count {
            cache.insert(make_batch(&[i as i32])).expect("insert");
        }
        let inner = DatasetInner {
            cache: cache.clone(),
            upstream: None,
            ingested_count: batch_count,
            upstream_exhausted: true,
            closed: false,
            max_readers,
            readers_created: 0,
            reader_positions: Vec::new(),
            evicted_up_to: 0,
        };
        (Arc::new(Mutex::new(inner)), cache)
    }

    #[test]
    fn register_reader_tracks_position() {
        let (inner_arc, _) = make_dataset_inner(Some(2), 4);
        let mut inner = inner_arc.lock().unwrap();
        let pos = inner.register_reader(0, true).expect("register");
        assert!(pos.is_some());
        assert_eq!(inner.readers_created, 1);
        assert_eq!(inner.reader_positions.len(), 1);
    }

    #[test]
    fn register_reader_none_max_returns_none_position() {
        let (inner_arc, _) = make_dataset_inner(None, 4);
        let mut inner = inner_arc.lock().unwrap();
        let pos = inner.register_reader(0, true).expect("register");
        assert!(pos.is_none());
        assert_eq!(inner.readers_created, 0);
    }

    #[test]
    fn register_reader_rejects_excess() {
        let (inner_arc, _) = make_dataset_inner(Some(1), 4);
        let mut inner = inner_arc.lock().unwrap();
        let _pos = inner.register_reader(0, true).expect("first ok");
        assert!(inner.register_reader(0, true).is_err());
    }

    #[test]
    fn register_reader_rejects_from_start_after_eviction() {
        let (inner_arc, _) = make_dataset_inner(Some(2), 4);
        let mut inner = inner_arc.lock().unwrap();
        inner.evicted_up_to = 1;
        assert!(inner.register_reader(0, true).is_err());
        assert!(inner.register_reader(1, false).is_ok());
    }

    #[test]
    fn register_reader_prunes_dead_refs() {
        let (inner_arc, _) = make_dataset_inner(Some(3), 4);
        let mut inner = inner_arc.lock().unwrap();
        let _pos1 = inner.register_reader(0, true).expect("r1");
        let pos2 = inner.register_reader(0, true).expect("r2");
        drop(pos2);
        let _pos3 = inner.register_reader(0, true).expect("r3");
        assert_eq!(inner.reader_positions.len(), 2);
    }

    #[test]
    fn compute_lwm_returns_none_without_max_readers() {
        let (inner_arc, _) = make_dataset_inner(None, 4);
        let mut inner = inner_arc.lock().unwrap();
        assert!(inner.compute_lwm().is_none());
    }

    #[test]
    fn compute_lwm_returns_zero_before_all_readers_created() {
        let (inner_arc, _) = make_dataset_inner(Some(2), 4);
        let mut inner = inner_arc.lock().unwrap();
        let _pos = inner.register_reader(0, true).expect("register");
        assert_eq!(inner.compute_lwm(), Some(0));
    }

    #[test]
    fn compute_lwm_returns_min_position() {
        let (inner_arc, _) = make_dataset_inner(Some(2), 4);
        let mut inner = inner_arc.lock().unwrap();
        let pos1 = inner.register_reader(0, true).expect("r1").unwrap();
        let pos2 = inner.register_reader(0, true).expect("r2").unwrap();
        pos1.store(3, Ordering::Release);
        pos2.store(1, Ordering::Release);
        assert_eq!(inner.compute_lwm(), Some(1));
    }

    #[test]
    fn compute_lwm_prunes_dead_readers() {
        let (inner_arc, _) = make_dataset_inner(Some(2), 4);
        let mut inner = inner_arc.lock().unwrap();
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
        let mut inner = inner_arc.lock().unwrap();
        let pos = inner.register_reader(0, true).expect("register").unwrap();
        pos.store(3, Ordering::Release);
        inner.maybe_evict().expect("evict");
        assert_eq!(inner.evicted_up_to, 3);
        assert!(cache.get(0).expect("ok").is_none());
        assert!(cache.get(2).expect("ok").is_none());
        assert!(cache.get(3).expect("ok").is_some());
    }

    #[test]
    fn maybe_evict_noop_without_max_readers() {
        let (inner_arc, cache) = make_dataset_inner(None, 4);
        let mut inner = inner_arc.lock().unwrap();
        inner.maybe_evict().expect("evict");
        assert_eq!(inner.evicted_up_to, 0);
        assert!(cache.get(0).expect("ok").is_some());
    }

    #[test]
    fn reader_drop_triggers_eviction() {
        let (inner_arc, cache) = make_dataset_inner(Some(1), 4);
        let pos = {
            let mut inner = inner_arc.lock().unwrap();
            inner.register_reader(0, true).expect("register").unwrap()
        };
        pos.store(4, Ordering::Release);
        let reader = StreamCacheReader {
            schema: Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)])),
            inner: inner_arc.clone(),
            current_index: 4,
            position: Some(pos),
        };
        drop(reader);
        let inner = inner_arc.lock().unwrap();
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
        let inner = cache.inner.lock().expect("lock");
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
}
