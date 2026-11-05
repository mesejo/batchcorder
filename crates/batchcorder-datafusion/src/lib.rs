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

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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
}

impl TeeTableProvider {
    pub fn new(cache: StreamCache) -> Self {
        Self {
            cache,
            stats: Arc::new(TeeStats::default()),
        }
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
             for tables that are re-read (re-run plans, recursive CTEs)"
        )),
        other => DataFusionError::from(other),
    }
}

#[async_trait]
impl TableProvider for TeeTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

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
        let partition: Arc<dyn PartitionStream> = Arc::new(TeePartition {
            schema: schema.clone(),
            cache: self.cache.clone(),
            reader: Mutex::new(Some(reader)),
            stats: self.stats.clone(),
        });
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

/// One scan's replay of the cache.  Holds the reader reserved at planning
/// time until the partition executes; dropping an unexecuted plan drops the
/// reader and releases its hold on eviction.
struct TeePartition {
    schema: SchemaRef,
    cache: StreamCache,
    reader: Mutex<Option<StreamCacheReader>>,
    stats: Arc<TeeStats>,
}

impl fmt::Debug for TeePartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeePartition")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl TeePartition {
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
        match self.take_reader() {
            Ok(mut reader) => {
                let tx = builder.tx();
                // Reads block on upstream I/O, disk, other readers' ingest or
                // (for Python upstreams) the GIL, so they must never run on an
                // async worker thread.
                builder.spawn_blocking(move || {
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
                });
            }
            Err(e) => builder.spawn_blocking(move || Err(e)),
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
