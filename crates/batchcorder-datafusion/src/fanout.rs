//! [`TeeFanoutRule`]: make tee scans that a plan re-executes safe to re-read.

use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::recursive_query::RecursiveQueryExec;
use datafusion::physical_plan::streaming::StreamingTableExec;

use crate::tee_cache_of;

/// A physical optimizer rule for hosts that link `batchcorder-datafusion`
/// directly (`datafusion-ffi` cannot carry optimizer rules).
///
/// `RecursiveQueryExec` re-executes its recursive term on every iteration.
/// A tee scan inside that term would re-read a sealed cache after the first
/// iteration evicted it, and fail.  This rule finds those scans in the final
/// physical plan and switches their caches to
/// [`Bounding::Unbounded`](batchcorder_core::Bounding::Unbounded), so every
/// batch stays replayable.  The whole cache switches, so scans of the same
/// table outside the recursive term stop evicting too.  Scans outside any
/// recursive term are left alone.
///
/// The rule only inspects the plan; it never rewrites it.  It runs whenever
/// a physical plan is created, including for `EXPLAIN`, so explaining a
/// recursive query already unbounds the caches it touches.
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
        unbound_rescanned(&plan, false)?;
        Ok(plan)
    }

    fn name(&self) -> &str {
        "tee_fanout"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Unbound the cache of every tee scan that runs inside a recursive term.
/// `rescanned` is true once the walk is inside one.
fn unbound_rescanned(plan: &Arc<dyn ExecutionPlan>, rescanned: bool) -> Result<()> {
    if let Some(recursive) = plan.downcast_ref::<RecursiveQueryExec>() {
        unbound_rescanned(recursive.static_term(), rescanned)?;
        return unbound_rescanned(recursive.recursive_term(), true);
    }
    if rescanned && let Some(streaming) = plan.downcast_ref::<StreamingTableExec>() {
        for partition in streaming.partitions() {
            if let Some(cache) = tee_cache_of(partition)? {
                cache.make_unbounded().map_err(|e| {
                    DataFusionError::Plan(format!(
                        "TeeFanoutRule cannot let a recursive query re-read a tee \
                         table: {e}. Register a new provider for this query"
                    ))
                })?;
            }
        }
    }
    for child in plan.children() {
        unbound_rescanned(child, rescanned)?;
    }
    Ok(())
}
