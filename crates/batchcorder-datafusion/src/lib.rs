//! A DataFusion [`TableProvider`] that *tees* one read-once Arrow stream to
//! every scan of the table in a query.
//!
//! [`TeeTableProvider`] wraps a [`batchcorder_core::StreamCache`].  Each
//! [`TableProvider::scan`] call — one per `TableScan` that survives logical
//! optimisation — creates one cache reader at planning time.  With
//! [`Bounding::Sealed`] that makes the reader fan-out exact by construction:
//! no reader count is guessed from SQL, a pruned branch never calls `scan`,
//! and a plan that is dropped unexecuted releases its readers.  Execution
//! seals the cache on the first poll, after which each batch is evicted as
//! soon as every scan has read past it.
//!
//! The physical plan is DataFusion's own [`StreamingTableExec`] with one
//! partition per scan, so projection, `LIMIT` pushdown and plan display come
//! from DataFusion.  Parallelism comes from the `RepartitionExec` DataFusion
//! puts above it; the cache itself replays a single ordered stream.
//!
//! # One query per provider
//!
//! A sealed cache serves the scans planned before its first poll.  Planning
//! the same provider into a second query after the first one has started
//! evicting fails with a clear error.  Register a fresh provider per query,
//! or use [`Bounding::Unbounded`] when the table must be re-read.
//!
//! # Recursive queries
//!
//! A recursive CTE re-executes its recursive term on every iteration, so a
//! scan inside it re-reads the stream after a sealed cache has evicted it.
//! Hosts that link this crate directly can add [`TeeFanoutRule`] to the
//! session's physical optimizer: it pins every tee scan inside a recursive
//! term at the start of the stream, so nothing is evicted while that plan is
//! alive.  Without it, such a query fails with a clear error instead.

mod fanout;

pub use fanout::TeeFanoutRule;

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use async_trait::async_trait;
use batchcorder_core::{Bounding, StreamCache, StreamCacheReader};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::error::ArrowError;
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::TableType;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::stream::RecordBatchReceiverStreamBuilder;
use datafusion::physical_plan::streaming::{PartitionStream, StreamingTableExec};
use tokio::runtime::Handle;

/// Batches buffered between the blocking reader task and the consumer.
/// Small on purpose: a deep buffer lets one scan run ahead of the others,
/// which holds back eviction.
const CHANNEL_CAPACITY: usize = 2;

/// Counters describing how DataFusion used a [`TeeTableProvider`].
#[derive(Debug, Default)]
pub struct TeeStats {
    scans_planned: AtomicU64,
    scans_executed: AtomicU64,
}

impl TeeStats {
    /// Number of `scan()` calls: one per physical scan node planned.
    pub fn scans_planned(&self) -> u64 {
        self.scans_planned.load(Ordering::Relaxed)
    }

    /// Number of scan partitions that started executing.
    pub fn scans_executed(&self) -> u64 {
        self.scans_executed.load(Ordering::Relaxed)
    }
}

/// A [`TableProvider`] over a [`StreamCache`] that gives every scan in a
/// query its own replay of the same read-once stream.
///
/// Build the cache with [`Bounding::Sealed`] for bounded memory; see the
/// crate docs for how the fan-out is counted.
pub struct TeeTableProvider {
    cache: StreamCache,
    stats: Arc<TeeStats>,
    runtime: Option<Handle>,
}

impl TeeTableProvider {
    pub fn new(cache: StreamCache) -> Self {
        Self {
            cache,
            stats: Arc::new(TeeStats::default()),
            runtime: None,
        }
    }

    /// Run the blocking reader tasks on `runtime` instead of the ambient
    /// Tokio runtime.  Required when the provider is used across
    /// `datafusion-ffi`: the host's runtime belongs to another copy of Tokio
    /// that this library cannot see.
    pub fn with_runtime(mut self, runtime: Handle) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// The underlying cache, e.g. to inspect `ingested_count` or
    /// `evicted_count`.
    pub fn cache(&self) -> &StreamCache {
        &self.cache
    }

    pub fn stats(&self) -> Arc<TeeStats> {
        self.stats.clone()
    }
}

impl fmt::Debug for TeeTableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeeTableProvider")
            .field("schema", &self.cache.schema())
            .field("stats", &self.stats)
            .field("runtime", &self.runtime.is_some())
            .finish()
    }
}

/// Map a reader-creation failure to an error that says what to do about it.
fn reader_error(e: ArrowError, during: &str) -> DataFusionError {
    match e {
        ArrowError::InvalidArgumentError(msg) => DataFusionError::Execution(format!(
            "TeeTableProvider cannot add a scan {during}: {msg}. A tee provider \
             serves the scans planned before its first execution; register a \
             new provider per query, or build the cache with Bounding::Unbounded \
             for tables that are re-read (re-run plans). For recursive CTEs, \
             hosts that link batchcorder-datafusion can add TeeFanoutRule to \
             the physical optimizer instead"
        )),
        other => DataFusionError::from(other),
    }
}

#[async_trait]
impl TableProvider for TeeTableProvider {
    fn schema(&self) -> SchemaRef {
        self.cache.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // The reservation: one reader per planned scan, created now so the
        // cache knows every consumer before execution polls any of them.
        let reader = self
            .cache
            .reader(true)
            .map_err(|e| reader_error(e, "while planning"))?;
        self.stats.scans_planned.fetch_add(1, Ordering::Relaxed);
        let schema = self.cache.schema();
        let partition = Arc::new(TeePartition {
            schema: schema.clone(),
            cache: self.cache.clone(),
            reader: Mutex::new(Some(reader)),
            pin: Mutex::new(None),
            stats: self.stats.clone(),
            runtime: self.runtime.clone(),
        });
        register_partition(&partition);
        let partition: Arc<dyn PartitionStream> = partition;
        let exec = StreamingTableExec::try_new(
            schema,
            vec![partition],
            projection,
            Vec::new(),
            false,
            limit,
        )?;
        Ok(Arc::new(exec))
    }
}

/// Every live [`TeePartition`], so that [`TeeFanoutRule`] can recognise tee
/// scans in a physical plan: DataFusion's [`PartitionStream`] cannot be
/// downcast.  Holding a `Weak` keeps the allocation (and so its address)
/// reserved, so a match by address is never a different, newer partition.
/// Only touched at planning time.
static PARTITIONS: Mutex<Vec<Weak<TeePartition>>> = Mutex::new(Vec::new());

/// The registry, even after a panic elsewhere poisoned its lock: entries are
/// only pushed and pruned, so the list is always valid.
fn partitions() -> MutexGuard<'static, Vec<Weak<TeePartition>>> {
    PARTITIONS.lock().unwrap_or_else(PoisonError::into_inner)
}

fn register_partition(partition: &Arc<TeePartition>) {
    let mut live = partitions();
    live.retain(|w| w.strong_count() > 0);
    live.push(Arc::downgrade(partition));
}

/// Live tee partitions keyed by address, taken under one lock.  Dead entries
/// are pruned on the way.
fn live_partitions() -> HashMap<usize, Arc<TeePartition>> {
    let mut live = partitions();
    live.retain(|w| w.strong_count() > 0);
    live.iter()
        .filter_map(|w| Some((Weak::as_ptr(w).addr(), w.upgrade()?)))
        .collect()
}

/// Key of `partition` in [`live_partitions`].
fn partition_key(partition: &Arc<dyn PartitionStream>) -> usize {
    Arc::as_ptr(partition).cast::<()>().addr()
}

/// One scan's replay of the cache.  Holds the reader reserved at planning
/// time until the partition executes; dropping an unexecuted plan drops the
/// reader and releases its hold on eviction.
struct TeePartition {
    schema: SchemaRef,
    cache: StreamCache,
    reader: Mutex<Option<StreamCacheReader>>,
    /// Set by [`TeeFanoutRule`] when the plan re-executes this partition: an
    /// unpolled from-start reader that keeps every batch replayable while the
    /// plan is alive.  Dropped with the plan, which lets eviction resume.
    pin: Mutex<Option<StreamCacheReader>>,
    stats: Arc<TeeStats>,
    runtime: Option<Handle>,
}

impl fmt::Debug for TeePartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeePartition")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl TeePartition {
    /// Hold the cache at the start of the stream for as long as this
    /// partition lives, so re-executions can read from the start.  Fails
    /// once anything has been evicted.  Idempotent.
    fn pin(&self) -> std::result::Result<(), ArrowError> {
        let mut pin = self.pin.lock().unwrap_or_else(PoisonError::into_inner);
        if pin.is_none() {
            *pin = Some(self.cache.reader(true)?);
        }
        Ok(())
    }

    /// The reader reserved at planning time, or a fresh from-start reader
    /// when this partition executes again (re-run plan, recursive CTE).  A
    /// fresh reader only succeeds while nothing has been evicted.
    fn take_reader(&self) -> Result<StreamCacheReader> {
        let reserved = self.reader.lock().ok().and_then(|mut slot| slot.take());
        match reserved {
            Some(reader) => Ok(reader),
            None => self
                .cache
                .reader(true)
                .map_err(|e| reader_error(e, "for a re-executed scan")),
        }
    }
}

impl PartitionStream for TeePartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        self.stats.scans_executed.fetch_add(1, Ordering::Relaxed);
        let mut builder =
            RecordBatchReceiverStreamBuilder::new(self.schema.clone(), CHANNEL_CAPACITY);
        // Reads block on upstream I/O, disk, other readers' ingest or (for
        // Python upstreams) the GIL, so they must never run on an async
        // worker thread.
        let task: Box<dyn FnOnce() -> Result<()> + Send> = match self.take_reader() {
            Ok(mut reader) => {
                let tx = builder.tx();
                Box::new(move || {
                    for item in reader.by_ref() {
                        let item = item.map_err(DataFusionError::from);
                        let failed = item.is_err();
                        // A closed channel means the consumer is done (query
                        // finished or LIMIT reached): stop, and drop the
                        // reader so it stops holding back eviction.
                        if tx.blocking_send(item).is_err() || failed {
                            break;
                        }
                    }
                    Ok(())
                })
            }
            Err(e) => Box::new(move || Err(e)),
        };
        match &self.runtime {
            Some(handle) => builder.spawn_blocking_on(task, handle),
            None => builder.spawn_blocking(task),
        }
        builder.build()
    }
}

/// Convenience for the common case: a sealed cache over `upstream`, wrapped
/// in a provider for one query.
pub fn sealed_provider(
    upstream: Box<dyn batchcorder_core::Upstream>,
    schema: SchemaRef,
    config: batchcorder_core::CacheConfig,
) -> Result<TeeTableProvider> {
    let config = batchcorder_core::CacheConfig {
        bounding: Bounding::Sealed,
        ..config
    };
    let cache = StreamCache::try_new(upstream, schema, config)?;
    Ok(TeeTableProvider::new(cache))
}
