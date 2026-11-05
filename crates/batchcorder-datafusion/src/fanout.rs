//! [`TeeFanoutRule`]: make tee scans that a plan re-executes safe to re-read.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::error::ArrowError;
use datafusion::common::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::recursive_query::RecursiveQueryExec;
use datafusion::physical_plan::streaming::StreamingTableExec;

use crate::{TeePartition, live_partitions, partition_key};

/// A physical optimizer rule for hosts that link `batchcorder-datafusion`
/// directly (`datafusion-ffi` cannot carry optimizer rules).
///
/// `RecursiveQueryExec` re-executes its recursive term on every iteration.
/// A tee scan inside that term would re-read a sealed cache after the first
/// iteration evicted it, and fail.  This rule finds those scans in the final
/// physical plan and *pins* each one: the scan holds an extra, never-polled
/// reader at the start of the stream, so the cache keeps every batch while
/// the plan is alive.  Scans of the same table elsewhere in the plan share
/// the cache, so they stop evicting too while the plan lives.
///
/// The pin belongs to the plan: when the plan is dropped (after execution,
/// after an `EXPLAIN`, or when planning fails), the pin goes with it and
/// eviction resumes.  Nothing about the cache changes permanently.  Scans
/// outside any recursive term are left alone, and the plan is never
/// rewritten.
///
/// ```ignore
/// let state = SessionStateBuilder::new()
///     .with_default_features()
///     .with_physical_optimizer_rule(Arc::new(TeeFanoutRule))
///     .build();
/// let ctx = SessionContext::new_with_state(state);
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct TeeFanoutRule;

impl PhysicalOptimizerRule for TeeFanoutRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // One registry lock per rule run.  If a pin fails, the error drops
        // the plan, and with it every pin taken so far.
        pin_rescanned(&plan, false, &live_partitions())?;
        Ok(plan)
    }

    fn name(&self) -> &str {
        "tee_fanout"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Pin every tee scan that runs inside a recursive term.  `rescanned` is
/// true once the walk is inside one.
fn pin_rescanned(
    plan: &Arc<dyn ExecutionPlan>,
    rescanned: bool,
    tee: &HashMap<usize, Arc<TeePartition>>,
) -> Result<()> {
    if let Some(recursive) = plan.downcast_ref::<RecursiveQueryExec>() {
        pin_rescanned(recursive.static_term(), rescanned, tee)?;
        return pin_rescanned(recursive.recursive_term(), true, tee);
    }
    if rescanned && let Some(streaming) = plan.downcast_ref::<StreamingTableExec>() {
        for partition in streaming.partitions() {
            if let Some(tee_partition) = tee.get(&partition_key(partition)) {
                tee_partition.pin().map_err(pin_error)?;
            }
        }
    }
    for child in plan.children() {
        pin_rescanned(child, rescanned, tee)?;
    }
    Ok(())
}

/// Only an eviction (or reader-cap) failure is the caller's to fix with a
/// new provider; anything else is passed through as is.
fn pin_error(e: ArrowError) -> DataFusionError {
    match e {
        ArrowError::InvalidArgumentError(msg) => DataFusionError::Plan(format!(
            "TeeFanoutRule cannot let a recursive query re-read a tee table: \
             {msg}. Register a new provider for this query"
        )),
        other => DataFusionError::from(other),
    }
}
