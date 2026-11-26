//! Probes of DuckDB's C-API table-function lifecycle.
//!
//! A counting table function, `probe()`, logs every `bind`, `init`, scan call
//! and bind-data drop, tagged with the id of the bind that produced it.  Each
//! probe runs one query shape and asserts the counts DuckDB 2.0.1 (bundled by
//! duckdb-rs 1.20001.0) was observed to produce.  They pin the facts the tee
//! design relies on, so a DuckDB upgrade that changes them fails here first.
//!
//! The probes are `#[ignore]`d: they document engine behaviour rather than
//! test this crate.  Run them with
//!
//! ```text
//! cargo test -p batchcorder-duckdb --test probes -- --ignored --test-threads=1
//! ```

use std::error::Error;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use duckdb::Connection;
use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};

/// Rows per scan chunk: DuckDB's standard vector size.
const CHUNK: u64 = 2048;
/// Chunks each scan emits unless `chunks := n` says otherwise.
const CHUNKS: u64 = 3;
const ROWS: u64 = CHUNK * CHUNKS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    Bind(u64),
    Init(u64),
    Func(u64),
    Drop(u64),
}

static EVENTS: Mutex<Vec<Event>> = Mutex::new(Vec::new());
static NEXT_BIND: AtomicU64 = AtomicU64::new(0);
/// Serializes probes: the event log is process-global.
static PROBE_LOCK: Mutex<()> = Mutex::new(());

fn log(event: Event) {
    EVENTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(event);
}

struct ProbeBind {
    id: u64,
    chunks: u64,
}

impl Drop for ProbeBind {
    fn drop(&mut self) {
        log(Event::Drop(self.id));
    }
}

struct ProbeInit {
    emitted: AtomicU64,
}

struct Probe;

impl VTab for Probe {
    type InitData = ProbeInit;
    type BindData = ProbeBind;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("id", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("m", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        let chunks = match bind.get_named_parameter("chunks") {
            Some(v) => u64::try_from(v.to_int64())?,
            None => CHUNKS,
        };
        let id = NEXT_BIND.fetch_add(1, Ordering::Relaxed);
        log(Event::Bind(id));
        Ok(ProbeBind { id, chunks })
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        init.set_max_threads(1);
        // SAFETY: DuckDB hands init the bind data this function's `bind`
        // returned; it outlives the init call.
        let bind = unsafe { &*init.get_bind_data::<ProbeBind>() };
        log(Event::Init(bind.id));
        Ok(ProbeInit {
            emitted: AtomicU64::new(0),
        })
    }

    fn named_parameters() -> Option<Vec<(String, LogicalTypeHandle)>> {
        Some(vec![(
            "chunks".into(),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
        )])
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let bind = func.get_bind_data();
        let chunk = func.get_init_data().emitted.fetch_add(1, Ordering::Relaxed);
        if chunk >= bind.chunks {
            output.set_len(0);
            return Ok(());
        }
        log(Event::Func(bind.id));
        let start = chunk * CHUNK;
        let values: Vec<i64> = (start..start + CHUNK).map(|v| v as i64).collect();
        for col in 0..2 {
            let mut vector = output.flat_vector(col);
            // SAFETY: both columns are BIGINT and CHUNK equals the vector
            // capacity.
            unsafe { vector.copy(&values) };
        }
        output.set_len(CHUNK as usize);
        Ok(())
    }
}

/// The events of one *bind pass*: a set of binds that are all dropped
/// before the next pass starts.  DuckDB binds every statement twice — once
/// while preparing it (that pass is dropped without an `init`), and again
/// when it executes — and re-binds on every execution of a prepared
/// statement, because C-API bind data opts out of the statement cache.
#[derive(Debug, Default)]
struct Pass {
    events: Vec<Event>,
}

impl Pass {
    fn bind_ids(&self) -> Vec<u64> {
        self.events
            .iter()
            .filter_map(|e| {
                if let Event::Bind(i) = e {
                    Some(*i)
                } else {
                    None
                }
            })
            .collect()
    }

    fn binds(&self) -> usize {
        self.bind_ids().len()
    }

    fn drops(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, Event::Drop(_)))
            .count()
    }

    /// Init count per bind, in bind order; 0 for a bind never initialised.
    fn inits_per_bind(&self) -> Vec<usize> {
        self.bind_ids()
            .into_iter()
            .map(|b| self.events.iter().filter(|e| **e == Event::Init(b)).count())
            .collect()
    }

    fn scanned(&self, bind: u64) -> bool {
        self.events.contains(&Event::Func(bind))
    }

    /// Binds whose scan never ran (pruned, or the pass was not executed).
    fn unscanned(&self) -> Vec<u64> {
        self.bind_ids()
            .into_iter()
            .filter(|b| !self.scanned(*b))
            .collect()
    }

    /// Whether every unscanned bind was dropped before the first scan call.
    fn pruned_dropped_before_first_scan(&self) -> bool {
        let first_scan = self
            .events
            .iter()
            .position(|e| matches!(e, Event::Func(_)))
            .unwrap_or(self.events.len());
        self.unscanned()
            .iter()
            .all(|b| self.events[..first_scan].contains(&Event::Drop(*b)))
    }

    /// Whether scan calls of two binds alternate: some bind scans again after
    /// another bind's scan started.
    fn interleaved(&self) -> bool {
        let funcs: Vec<u64> = self
            .events
            .iter()
            .filter_map(|e| {
                if let Event::Func(i) = e {
                    Some(*i)
                } else {
                    None
                }
            })
            .collect();
        let mut finished = Vec::new();
        for pair in funcs.windows(2) {
            if pair[0] != pair[1] {
                if finished.contains(&pair[1]) {
                    return true;
                }
                finished.push(pair[0]);
            }
        }
        false
    }
}

/// Split a trace into bind passes.
fn passes(events: Vec<Event>) -> Vec<Pass> {
    let mut out = Vec::new();
    let mut current = Pass::default();
    let mut live = 0_i64;
    for e in events {
        match e {
            Event::Bind(_) => live += 1,
            Event::Drop(_) => live -= 1,
            _ => {}
        }
        current.events.push(e);
        if live == 0 {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.events.is_empty() {
        out.push(current);
    }
    out
}

struct Probing {
    con: Connection,
    _guard: MutexGuard<'static, ()>,
}

impl Probing {
    fn new(threads: u32) -> Self {
        let guard = PROBE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let con = Connection::open_in_memory().expect("open in-memory DuckDB");
        con.register_table_function::<Probe>("probe")
            .expect("register probe()");
        con.execute_batch(&format!("SET threads = {threads}"))
            .expect("set threads");
        Self { con, _guard: guard }
    }

    /// Run `f` and split what it did to `probe()` into bind passes,
    /// including the drops that happen once its statement is gone.
    fn trace<T>(&self, f: impl FnOnce(&Connection) -> T) -> (T, Vec<Pass>) {
        EVENTS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        let out = f(&self.con);
        let events = std::mem::take(&mut *EVENTS.lock().unwrap_or_else(PoisonError::into_inner));
        eprintln!("{events:?}");
        (out, passes(events))
    }

    /// Run a one-row, one-column query.  Returns its value, the planning
    /// pass and the execution pass.
    fn count(&self, sql: &str) -> (i64, Pass, Pass) {
        let (n, mut passes) = self.trace(|con| {
            con.query_row(sql, [], |r| r.get::<_, i64>(0))
                .unwrap_or_else(|e| panic!("{sql}: {e}"))
        });
        assert_eq!(
            passes.len(),
            2,
            "{sql}: expected a planning and an execution pass"
        );
        let exec = passes.pop().expect("two passes");
        let plan = passes.pop().expect("two passes");
        assert_planning_only(&plan, sql);
        (n, plan, exec)
    }
}

/// The planning pass binds and drops; it never initialises or scans.
fn assert_planning_only(plan: &Pass, what: &str) {
    assert!(
        plan.inits_per_bind().iter().all(|&n| n == 0),
        "{what}: planning pass initialised a scan: {plan:?}"
    );
    assert_eq!(
        plan.drops(),
        plan.binds(),
        "{what}: planning pass drops every bind"
    );
}

// ── P1: binds, inits and drops per query shape ──────────────────────────────
//
// One bind per table-function reference, one init per bind, one drop per
// bind: a reader created in `bind` matches the scans exactly.

#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p1_self_join() {
    let p = Probing::new(4);
    let (n, plan, exec) = p.count("SELECT count(*) FROM probe() a JOIN probe() b ON a.m = b.id");
    assert_eq!(n, ROWS as i64);
    assert_eq!(plan.binds(), 2);
    assert_eq!(exec.inits_per_bind(), vec![1, 1]);
    assert_eq!(exec.drops(), 2);
}

#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p1_union_all() {
    let p = Probing::new(4);
    let (n, _, exec) =
        p.count("SELECT count(*) FROM (SELECT * FROM probe() UNION ALL SELECT * FROM probe())");
    assert_eq!(n, 2 * ROWS as i64);
    assert_eq!(exec.inits_per_bind(), vec![1, 1]);
    assert_eq!(exec.drops(), 2);
}

#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p1_in_subquery() {
    let p = Probing::new(4);
    let (n, _, exec) =
        p.count("SELECT count(*) FROM probe() WHERE id IN (SELECT m FROM probe() WHERE m > 100)");
    assert_eq!(n, ROWS as i64 - 101);
    assert_eq!(exec.inits_per_bind(), vec![1, 1]);
    assert_eq!(exec.drops(), 2);
}

/// A CTE referenced twice is materialized by default: its body binds once
/// and is scanned once.  `NOT MATERIALIZED` binds the body once more, unused,
/// then once per inlined reference; the unused bind is dropped before any
/// scan runs.
#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p1_cte_twice() {
    let p = Probing::new(4);
    for (kind, inits) in [
        ("", vec![1]),
        ("MATERIALIZED", vec![1]),
        ("NOT MATERIALIZED", vec![0, 1, 1]),
    ] {
        let sql = format!(
            "WITH c AS {kind} (SELECT * FROM probe()) \
             SELECT count(*) FROM c x JOIN c y USING (id)"
        );
        let (n, _, exec) = p.count(&sql);
        assert_eq!(n, ROWS as i64, "{kind}");
        assert_eq!(exec.inits_per_bind(), inits, "{kind}: inits");
        assert_eq!(exec.drops(), inits.len(), "{kind}: drops");
        assert!(exec.pruned_dropped_before_first_scan(), "{kind}: {exec:?}");
    }
}

#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p1_asof_join() {
    let p = Probing::new(4);
    let (n, _, exec) =
        p.count("SELECT count(*) FROM probe() a ASOF JOIN probe() b ON a.id >= b.id");
    assert_eq!(n, ROWS as i64);
    assert_eq!(exec.inits_per_bind(), vec![1, 1]);
    assert_eq!(exec.drops(), 2);
}

// ── P2: pruned scans ────────────────────────────────────────────────────────
//
// The optimizer frees a pruned scan's bind data before any scan runs, so a
// reader held by that bind data stops holding back eviction in time.

#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p2_pruned_branch() {
    let p = Probing::new(4);
    let (n, _, exec) = p.count(
        "SELECT count(*) FROM (SELECT * FROM probe() UNION ALL SELECT * FROM probe() WHERE false)",
    );
    assert_eq!(n, ROWS as i64);
    assert_eq!(exec.binds(), 2);
    assert_eq!(exec.unscanned().len(), 1);
    assert!(exec.pruned_dropped_before_first_scan(), "{exec:?}");
}

#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p2_limit_zero() {
    let p = Probing::new(4);
    let (n, _, exec) = p.count("SELECT count(*) FROM (SELECT * FROM probe() LIMIT 0)");
    assert_eq!(n, 0);
    assert_eq!(exec.binds(), 1);
    assert_eq!(exec.unscanned().len(), 1);
    assert_eq!(exec.inits_per_bind(), vec![0]);
}

// ── P3: statements that bind without running, or run more than once ─────────

/// `EXPLAIN` and `DESCRIBE` bind (twice, like any statement) and never init.
#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p3_explain_and_describe() {
    let p = Probing::new(4);
    for sql in [
        "EXPLAIN SELECT * FROM probe() a JOIN probe() b ON a.m = b.id",
        "DESCRIBE SELECT * FROM probe()",
    ] {
        let ((), passes) = p.trace(|con| {
            let mut stmt = con.prepare(sql).expect("prepare");
            let mut rows = stmt.query([]).expect("query");
            while rows.next().expect("row").is_some() {}
        });
        assert_eq!(passes.len(), 2, "{sql}");
        for pass in &passes {
            assert_planning_only(pass, sql);
        }
    }
}

/// A prepared statement re-binds on every execution: each `EXECUTE` gets
/// fresh bind data, and so fresh readers.
#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p3_prepare_execute_twice() {
    let p = Probing::new(4);
    let sql = "SELECT count(*) FROM probe() a JOIN probe() b ON a.m = b.id";
    let ((), passes) = p.trace(|con| {
        let mut stmt = con.prepare(sql).expect("prepare");
        for _ in 0..2 {
            let n: i64 = stmt.query_row([], |r| r.get(0)).expect("execute");
            assert_eq!(n, ROWS as i64);
        }
    });
    assert_eq!(passes.len(), 3, "prepare, then one pass per execute");
    assert_planning_only(&passes[0], sql);
    for exec in &passes[1..] {
        assert_eq!(exec.inits_per_bind(), vec![1, 1]);
        assert_eq!(exec.drops(), 2);
    }
}

// ── P4: shapes that may re-run a scan ───────────────────────────────────────

/// A recursive CTE re-initialises the scan in its recursive term once per
/// iteration, sequentially, on the same bind data.  The tee needs a re-init
/// path (a fresh from-start reader) and cannot evict under such a scan.
#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p4_recursive_cte() {
    let p = Probing::new(4);
    let (n, _, exec) = p.count(
        "WITH RECURSIVE r(n) AS (SELECT 0 UNION ALL \
           SELECT r.n + 1 FROM r JOIN probe() p ON p.id = r.n WHERE r.n < 3) \
         SELECT count(*) FROM r",
    );
    assert_eq!(n, 4);
    assert_eq!(exec.inits_per_bind(), vec![4]);
    assert_eq!(exec.drops(), 1);
}

/// A correlated (`LATERAL`) subquery is decorrelated: one bind, one init.
#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p4_correlated_subquery() {
    let p = Probing::new(4);
    let (n, _, exec) = p.count(
        "SELECT count(*) FROM range(5) t(x), \
         LATERAL (SELECT count(*) AS c FROM probe() p WHERE p.id < t.x)",
    );
    assert_eq!(n, 5);
    assert_eq!(exec.inits_per_bind(), vec![1]);
}

// ── P5: do two single-threaded scans run side by side? ──────────────────────

#[test]
#[ignore = "probe of DuckDB behaviour; run with --ignored"]
fn p5_scan_order() {
    let p = Probing::new(4);
    let (_, _, exec) = p.count(
        "SELECT count(*) FROM (SELECT * FROM probe(chunks := 200) \
         UNION ALL SELECT * FROM probe(chunks := 200))",
    );
    eprintln!("union all interleaved: {}", exec.interleaved());
    // A hash join reads its build side to the end before the probe side
    // starts: a tee must keep the whole stream (memory or disk).
    let (_, _, exec) = p.count(
        "SELECT count(*) FROM probe(chunks := 200) a JOIN probe(chunks := 200) b ON a.m = b.id",
    );
    assert!(
        !exec.interleaved(),
        "hash join scans run one after the other"
    );
}
