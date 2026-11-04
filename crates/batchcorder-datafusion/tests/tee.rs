//! End-to-end tests: SQL over a `TeeTableProvider` must match the same SQL
//! over a `MemTable` holding the same data, while planning exactly one
//! reader per surviving scan and evicting everything once the query is done.

use std::sync::Arc;

use batchcorder_core::{Bounding, CacheConfig, StreamCache};
use batchcorder_datafusion::{TeeFanoutRule, TeeStats, TeeTableProvider, sealed_provider};
use datafusion::arrow::array::{Int64Array, RecordBatch};
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::error::ArrowError;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;

const N_BATCHES: i64 = 50;
const ROWS: i64 = 20;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("x", DataType::Int64, false),
    ]))
}

fn batches() -> Vec<RecordBatch> {
    (0..N_BATCHES)
        .map(|b| {
            let x: Vec<i64> = (b * ROWS..(b + 1) * ROWS).collect();
            let k: Vec<i64> = x.iter().map(|v| v % 7).collect();
            RecordBatch::try_new(
                schema(),
                vec![Arc::new(Int64Array::from(k)), Arc::new(Int64Array::from(x))],
            )
            .expect("batch")
        })
        .collect()
}

fn upstream() -> Box<dyn batchcorder_core::Upstream> {
    let items: Vec<Result<RecordBatch, ArrowError>> = batches().into_iter().map(Ok).collect();
    Box::new(items.into_iter())
}

/// A context with `t` backed by a fresh sealed tee, plus handles to inspect it.
fn tee_ctx(config: CacheConfig) -> (SessionContext, StreamCache, Arc<TeeStats>) {
    let provider = sealed_provider(upstream(), schema(), config).expect("provider");
    let cache = provider.cache().clone();
    let stats = provider.stats();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider))
        .expect("register");
    (ctx, cache, stats)
}

/// The same query over an in-memory copy of the data.
async fn reference(sql: &str) -> RecordBatch {
    let ctx = SessionContext::new();
    let table = MemTable::try_new(schema(), vec![batches()]).expect("memtable");
    ctx.register_table("t", Arc::new(table)).expect("register");
    run(&ctx, sql).await.expect("reference query")
}

async fn run(ctx: &SessionContext, sql: &str) -> datafusion::error::Result<RecordBatch> {
    let df = ctx.sql(sql).await?;
    let schema = Arc::new(df.schema().as_arrow().clone());
    let out = df.collect().await?;
    Ok(concat_batches(&schema, &out)?)
}

/// Run `sql` on a fresh tee, compare with the reference, and check the
/// fan-out and that every batch was evicted afterwards.
async fn check(sql: &str, expected_scans: u64) {
    let (ctx, cache, stats) = tee_ctx(CacheConfig::default());
    let got = run(&ctx, sql).await.expect("tee query");
    assert_eq!(got, reference(sql).await, "result mismatch for: {sql}");
    assert_eq!(
        stats.scans_planned(),
        expected_scans,
        "scans planned for: {sql}"
    );
    assert_eq!(
        stats.scans_executed(),
        expected_scans,
        "scans executed for: {sql}"
    );
    let ingested = cache.ingested_count().expect("ok");
    assert_eq!(
        cache.evicted_count().expect("ok"),
        ingested,
        "not all evicted: {sql}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_scan() {
    check("select count(*) as c, sum(x) as s from t", 1).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn projection_and_filter() {
    check("select x from t where k = 3 order by x", 1).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn self_join() {
    check(
        "select a.k, count(*) as c from t a join t b on a.k = b.k group by a.k order by a.k",
        2,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn union_all_with_itself() {
    check(
        "select sum(x) as s, count(*) as c from (select x from t union all select x from t)",
        2,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cte_referenced_three_times() {
    check(
        "with c as (select k, x from t where x % 2 = 0) \
         select (select count(*) from c) as n, \
                (select sum(x) from c) as s, \
                (select max(k) from c) as m",
        3,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn in_subquery() {
    check(
        "select count(*) as c from t where x in (select x * 2 from t where k = 1)",
        2,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pruned_branch_plans_no_scan() {
    // The optimiser removes the `where false` branch before physical
    // planning, so only one scan reserves a reader.
    check(
        "select count(*) as c from (select x from t union all select x from t where false)",
        1,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn limit_on_one_side_releases_its_reader() {
    check(
        "select a.x, b.c from (select x from t order by x limit 1) a \
         cross join (select count(*) as c from t) b",
        2,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explain_then_execute_on_same_provider() {
    let (ctx, cache, stats) = tee_ctx(CacheConfig::default());
    let sql = "select count(*) as c from t a join t b on a.k = b.k";
    run(&ctx, &format!("explain {sql}")).await.expect("explain");
    // Planning reserved readers but nothing was pulled, and the dropped plan
    // released them, so the real query can still read from the start.
    assert_eq!(cache.ingested_count().expect("ok"), 0);
    assert!(!cache.is_sealed().expect("ok"));
    let got = run(&ctx, sql).await.expect("query");
    assert_eq!(got, reference(sql).await);
    assert_eq!(stats.scans_planned(), 4, "2 for EXPLAIN, 2 for the query");
    assert_eq!(stats.scans_executed(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropped_physical_plan_releases_readers() {
    let (ctx, cache, _stats) = tee_ctx(CacheConfig::default());
    let plan = ctx
        .sql("select * from t a join t b on a.x = b.x")
        .await
        .expect("sql")
        .create_physical_plan()
        .await
        .expect("plan");
    drop(plan);
    let sql = "select sum(x) as s from t";
    assert_eq!(run(&ctx, sql).await.expect("query"), reference(sql).await);
    assert_eq!(
        cache.evicted_count().expect("ok"),
        cache.ingested_count().expect("ok")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn second_query_after_eviction_fails_clearly() {
    let (ctx, _cache, _stats) = tee_ctx(CacheConfig::default());
    let df = ctx.sql("select sum(x) from t").await.expect("sql");
    df.clone().collect().await.expect("first run");
    let err = df.collect().await.expect_err("second run must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("register a new provider"),
        "unexpected error: {msg}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unbounded_cache_can_be_queried_repeatedly() {
    let cache = StreamCache::try_new(
        upstream(),
        schema(),
        CacheConfig {
            bounding: Bounding::Unbounded,
            ..CacheConfig::default()
        },
    )
    .expect("cache");
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(TeeTableProvider::new(cache)))
        .expect("register");
    let sql = "select k, sum(x) as s from t group by k order by k";
    let expected = reference(sql).await;
    for _ in 0..3 {
        assert_eq!(run(&ctx, sql).await.expect("query"), expected);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_scan_runs_in_bounded_memory() {
    // A memory budget of about two batches: only works if each batch is
    // evicted as soon as the (single) scan has read past it.
    let batch_bytes: usize = batches()[0]
        .columns()
        .iter()
        .map(|c| c.get_array_memory_size())
        .sum();
    let (ctx, cache, _stats) = tee_ctx(CacheConfig {
        memory_capacity: Some(batch_bytes * 2),
        ..CacheConfig::default()
    });
    let sql = "select count(*) as c, sum(x) as s from t";
    assert_eq!(run(&ctx, sql).await.expect("query"), reference(sql).await);
    assert_eq!(cache.ingested_count().expect("ok"), N_BATCHES as u64);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disk_tier_self_join() {
    let dir = std::env::temp_dir().join("batchcorder_datafusion_disk");
    let (ctx, cache, stats) = tee_ctx(CacheConfig {
        // Tiny hot layer: most reads come back from the IPC file.
        memory_capacity: Some(1),
        disk: Some(batchcorder_core::DiskConfig {
            path: dir,
            capacity: 1 << 26,
        }),
        ..CacheConfig::default()
    });
    let sql = "select a.k, count(*) as c from t a join t b on a.x = b.x group by a.k order by a.k";
    assert_eq!(run(&ctx, sql).await.expect("query"), reference(sql).await);
    assert_eq!(stats.scans_planned(), 2);
    cache.close().expect("close");
}

// ── recursive queries and TeeFanoutRule ──────────────────────────────────────

/// The tee scan sits in the recursive term, so it re-executes per iteration.
const RECURSIVE_SCAN_SQL: &str = "with recursive r(n) as (select 0 as n union all \
     select r.n + 1 from r join (select max(k) as m from t) s on r.n < s.m) \
     select n from r order by n";

/// Like [`tee_ctx`], with [`TeeFanoutRule`] in the physical optimizer.
fn tee_ctx_with_rule() -> (SessionContext, StreamCache, Arc<TeeStats>) {
    let provider = sealed_provider(upstream(), schema(), CacheConfig::default()).expect("provider");
    let cache = provider.cache().clone();
    let stats = provider.stats();
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(TeeFanoutRule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t", Arc::new(provider))
        .expect("register");
    (ctx, cache, stats)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recursive_scan_without_rule_fails_clearly() {
    let (ctx, _cache, _stats) = tee_ctx(CacheConfig::default());
    let err = run(&ctx, RECURSIVE_SCAN_SQL)
        .await
        .expect_err("second iteration re-reads an evicted cache");
    assert!(
        err.to_string().contains("Bounding::Unbounded"),
        "unexpected error: {err}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rule_makes_recursive_scan_rereadable() {
    let (ctx, cache, stats) = tee_ctx_with_rule();
    let got = run(&ctx, RECURSIVE_SCAN_SQL).await.expect("tee query");
    assert_eq!(got, reference(RECURSIVE_SCAN_SQL).await);
    assert_eq!(got.num_rows(), 7, "max(k) = 6, so n runs 0..=6");
    assert_eq!(cache.bounding().expect("ok"), Bounding::Unbounded);
    assert_eq!(cache.evicted_count().expect("ok"), 0);
    // One planned scan, re-executed once per iteration that reaches it.
    assert_eq!(stats.scans_planned(), 1);
    assert!(stats.scans_executed() > 1, "{}", stats.scans_executed());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rule_leaves_scan_in_static_term_sealed() {
    let sql = "with recursive r(n) as (select min(x) as n from t union all \
               select n + 1 from r where n < 5) select n from r order by n";
    let (ctx, cache, stats) = tee_ctx_with_rule();
    assert_eq!(
        run(&ctx, sql).await.expect("tee query"),
        reference(sql).await
    );
    assert_eq!(cache.bounding().expect("ok"), Bounding::Sealed);
    assert_eq!(stats.scans_executed(), 1);
    assert_eq!(
        cache.evicted_count().expect("ok"),
        cache.ingested_count().expect("ok")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rule_leaves_non_recursive_query_sealed() {
    let sql = "select a.k, count(*) as c from t a join t b on a.k = b.k group by a.k order by a.k";
    let (ctx, cache, stats) = tee_ctx_with_rule();
    assert_eq!(
        run(&ctx, sql).await.expect("tee query"),
        reference(sql).await
    );
    assert_eq!(cache.bounding().expect("ok"), Bounding::Sealed);
    assert_eq!(stats.scans_planned(), 2);
    assert_eq!(
        cache.evicted_count().expect("ok"),
        cache.ingested_count().expect("ok")
    );
}
