//! End-to-end tests of `tee_register` / `tee_scan` / `tee_drop` against an
//! in-process DuckDB.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::builder::{Int32Builder, ListBuilder, StringDictionaryBuilder};
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::types::Int32Type;
use arrow_array::{
    ArrayRef, Int32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray, StructArray,
};
use arrow_schema::{DataType, Field, Schema};
use duckdb::Connection;

/// A read-once upstream over `batches` that counts how many it hands out.
struct Upstream {
    stream: Box<FFI_ArrowArrayStream>,
    pulls: Arc<AtomicUsize>,
}

impl Upstream {
    fn new(batches: Vec<RecordBatch>) -> Self {
        let schema = batches[0].schema();
        let pulls = Arc::new(AtomicUsize::new(0));
        let counter = pulls.clone();
        let iter = batches.into_iter().map(move |b| {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(b)
        });
        let reader = RecordBatchIterator::new(iter, schema);
        Self {
            stream: Box::new(FFI_ArrowArrayStream::new(Box::new(reader))),
            pulls,
        }
    }

    fn ptr(&mut self) -> u64 {
        std::ptr::from_mut(self.stream.as_mut()) as u64
    }

    fn pulls(&self) -> usize {
        self.pulls.load(Ordering::Relaxed)
    }
}

fn connect() -> Connection {
    let con = Connection::open_in_memory().expect("open in-memory DuckDB");
    batchcorder_duckdb::register(&con).expect("register tee functions");
    con.execute_batch("SET threads = 4").expect("set threads");
    con
}

fn register(con: &Connection, name: &str, upstream: &mut Upstream) -> (String, u64) {
    let sql = format!("CALL tee_register('{name}', {}::UBIGINT)", upstream.ptr());
    con.query_row(&sql, [], |r| Ok((r.get(0)?, r.get(1)?)))
        .expect("tee_register")
}

/// `n` batches of `rows` rows each, `id` and `m` both counting up from 0.
fn id_batches(n: usize, rows: usize) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("m", DataType::Int64, false),
    ]));
    (0..n)
        .map(|i| {
            let start = (i * rows) as i64;
            let values: ArrayRef =
                Arc::new(Int64Array::from_iter_values(start..start + rows as i64));
            RecordBatch::try_new(schema.clone(), vec![values.clone(), values]).expect("batch")
        })
        .collect()
}

fn rows(con: &Connection, sql: &str) -> Vec<(i64, i64)> {
    let mut stmt = con.prepare(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .and_then(Iterator::collect)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// The same query over the tee and over a native table holding the same
/// rows.
fn assert_matches_reference(con: &Connection, query: &str, total: usize) {
    con.execute_batch(&format!(
        "CREATE OR REPLACE TABLE reference AS SELECT range AS id, range AS m FROM range({total})"
    ))
    .expect("reference table");
    let tee = rows(con, &query.replace("{t}", "tee_scan('t')"));
    let reference = rows(con, &query.replace("{t}", "reference"));
    assert!(!reference.is_empty());
    assert_eq!(tee, reference);
}

#[test]
fn self_join_reads_upstream_once() {
    let con = connect();
    let batches = id_batches(10, 1000);
    let mut upstream = Upstream::new(batches);
    assert_eq!(register(&con, "t", &mut upstream), ("t".into(), 2));

    assert_matches_reference(
        &con,
        "SELECT a.id, b.m FROM {t} a JOIN {t} b ON a.m = b.id ORDER BY a.id",
        10_000,
    );
    assert_eq!(
        upstream.pulls(),
        10,
        "every batch pulled from upstream once"
    );
}

#[test]
fn batch_larger_than_a_vector_is_emitted_in_slices() {
    let con = connect();
    let mut upstream = Upstream::new(id_batches(1, 5000));
    register(&con, "t", &mut upstream);
    assert_matches_reference(&con, "SELECT id, m FROM {t} ORDER BY id", 5000);
}

#[test]
fn registering_takes_the_stream() {
    let con = connect();
    let mut upstream = Upstream::new(id_batches(1, 10));
    assert!(upstream.stream.release.is_some());
    register(&con, "t", &mut upstream);
    assert!(
        upstream.stream.release.is_none(),
        "the source is left released so the caller cannot release it again"
    );
}

#[test]
fn tee_drop_forgets_the_name() {
    let con = connect();
    let mut upstream = Upstream::new(id_batches(1, 10));
    register(&con, "t", &mut upstream);

    let dropped: (String, bool) = con
        .query_row("CALL tee_drop('t')", [], |r| Ok((r.get(0)?, r.get(1)?)))
        .expect("tee_drop");
    assert_eq!(dropped, ("t".into(), true));
    let err = con
        .query_row("SELECT count(*) FROM tee_scan('t')", [], |r| {
            r.get::<_, i64>(0)
        })
        .expect_err("scan of a dropped tee");
    assert!(err.to_string().contains("no tee named 't'"), "{err}");

    let again: bool = con
        .query_row("CALL tee_drop('t')", [], |r| r.get(1))
        .expect("tee_drop");
    assert!(!again);
}

/// `i, s, l, st.a, d` as read back by `bridge_carries_...`.
type Row = (i64, Option<String>, Option<String>, Option<i32>, String);

/// Every column type crosses the arrow 59 -> 60 bridge and reaches DuckDB
/// with its values.
#[test]
fn bridge_carries_primitive_nested_and_dictionary_columns() {
    let mut list = ListBuilder::new(Int32Builder::new());
    list.append_value([Some(1), Some(2)]);
    list.append_null();
    list.append_value([Some(3)]);
    let mut dict = StringDictionaryBuilder::<Int32Type>::new();
    dict.append_value("x");
    dict.append_value("y");
    dict.append_value("x");
    let point = StructArray::from(vec![(
        Arc::new(Field::new("a", DataType::Int32, true)),
        Arc::new(Int32Array::from(vec![Some(10), None, Some(30)])) as ArrayRef,
    )]);
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("i", Arc::new(Int64Array::from(vec![1, 2, 3]))),
        (
            "s",
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
        ),
        ("l", Arc::new(list.finish())),
        ("st", Arc::new(point)),
        ("d", Arc::new(dict.finish())),
    ];
    let batch = RecordBatch::try_from_iter(columns).expect("batch");

    let con = connect();
    let mut upstream = Upstream::new(vec![batch]);
    register(&con, "t", &mut upstream);

    let mut stmt = con
        .prepare("SELECT i, s, l::VARCHAR, st.a, d::VARCHAR FROM tee_scan('t') ORDER BY i")
        .expect("prepare");
    let got: Vec<Row> = stmt
        .query_map([], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
        })
        .and_then(Iterator::collect)
        .expect("rows");
    assert_eq!(
        got,
        vec![
            (
                1,
                Some("a".into()),
                Some("[1, 2]".into()),
                Some(10),
                "x".into()
            ),
            (2, None, None, None, "y".into()),
            (
                3,
                Some("c".into()),
                Some("[3]".into()),
                Some(30),
                "x".into()
            ),
        ]
    );
}

#[test]
fn tee_register_rejects_a_duplicate_name() {
    let con = connect();
    let mut first = Upstream::new(id_batches(1, 10));
    register(&con, "t", &mut first);
    let mut second = Upstream::new(id_batches(1, 10));
    let sql = format!("CALL tee_register('t', {}::UBIGINT)", second.ptr());
    let err = con.execute_batch(&sql).expect_err("duplicate name");
    assert!(err.to_string().contains("already exists"), "{err}");
    assert!(
        second.stream.release.is_none(),
        "the stream is taken even when registration fails"
    );
}

#[test]
fn tee_register_rejects_a_null_or_consumed_stream() {
    let con = connect();
    let err = con
        .execute_batch("CALL tee_register('t', 0::UBIGINT)")
        .expect_err("NULL stream_ptr");
    assert!(err.to_string().contains("stream_ptr is NULL"), "{err}");

    let mut upstream = Upstream::new(id_batches(1, 10));
    register(&con, "t", &mut upstream);
    let sql = format!("CALL tee_register('u', {}::UBIGINT)", upstream.ptr());
    let err = con.execute_batch(&sql).expect_err("consumed stream");
    assert!(err.to_string().contains("already consumed"), "{err}");
}

#[test]
fn null_name_is_an_error() {
    let con = connect();
    let mut upstream = Upstream::new(id_batches(1, 10));
    let sql = format!("CALL tee_register(NULL, {}::UBIGINT)", upstream.ptr());
    let err = con.execute_batch(&sql).expect_err("NULL name");
    assert!(err.to_string().contains("name is NULL"), "{err}");
    assert!(upstream.stream.release.is_some(), "the stream is not taken");
    for sql in ["SELECT * FROM tee_scan(NULL)", "CALL tee_drop(NULL)"] {
        let err = con.execute_batch(sql).expect_err(sql);
        assert!(err.to_string().contains("name is NULL"), "{sql}: {err}");
    }
}

/// List and dictionary columns nested in a batch larger than a vector: each
/// chunk carries its own rows' list values, and nested dictionaries are
/// decoded.
#[test]
fn nested_columns_in_a_batch_larger_than_a_vector() {
    let total = 5000;
    let mut list = ListBuilder::new(Int32Builder::new());
    let mut dicts = ListBuilder::new(StringDictionaryBuilder::<Int32Type>::new());
    for i in 0..total {
        list.append_value([Some(i), Some(i + 1)]);
        dicts
            .values()
            .append_value(if i % 2 == 0 { "even" } else { "odd" });
        dicts.append(true);
    }
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("i", Arc::new(Int32Array::from_iter_values(0..total))),
        ("l", Arc::new(list.finish())),
        ("d", Arc::new(dicts.finish())),
    ];
    let batch = RecordBatch::try_from_iter(columns).expect("batch");

    let con = connect();
    let mut upstream = Upstream::new(vec![batch]);
    // Read twice below.
    register_bounded(&con, "t", &mut upstream, "unbounded");

    let bad: i64 = con
        .query_row(
            "SELECT count(*) FROM tee_scan('t') \
             WHERE l <> [i, i + 1] \
                OR d[1] <> CASE WHEN i % 2 = 0 THEN 'even' ELSE 'odd' END",
            [],
            |r| r.get(0),
        )
        .expect("scan");
    assert_eq!(bad, 0);
    let count: i64 = con
        .query_row("SELECT count(*) FROM tee_scan('t')", [], |r| r.get(0))
        .expect("count");
    assert_eq!(count, i64::from(total));
}

// ── Sealed and unbounded tees ───────────────────────────────────────────────

/// `tee_register` with an explicit `bounding`.
fn register_bounded(con: &Connection, name: &str, upstream: &mut Upstream, bounding: &str) {
    let sql = format!(
        "CALL tee_register('{name}', {}::UBIGINT, bounding := '{bounding}')",
        upstream.ptr()
    );
    con.execute_batch(&sql).expect("tee_register");
}

/// `(ingested, evicted, sealed)` from `tee_stats`.
fn stats(con: &Connection, name: &str) -> (u64, u64, bool) {
    con.query_row(&format!("CALL tee_stats('{name}')"), [], |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?))
    })
    .expect("tee_stats")
}

fn query_error(con: &Connection, sql: &str) -> String {
    con.query_row(sql, [], |r| r.get::<_, i64>(0))
        .expect_err(sql)
        .to_string()
}

const UNION_ALL: &str =
    "SELECT id, m FROM (SELECT * FROM {t} UNION ALL SELECT * FROM {t}) ORDER BY id";

#[test]
fn sealed_is_the_default() {
    let con = connect();
    let mut upstream = Upstream::new(id_batches(2, 10));
    register(&con, "t", &mut upstream);
    assert_eq!(stats(&con, "t"), (0, 0, false));
    con.query_row("SELECT count(*) FROM tee_scan('t')", [], |r| {
        r.get::<_, i64>(0)
    })
    .expect("scan");
    assert!(stats(&con, "t").2, "the first read seals the cache");
}

/// Two scans read side by side; each batch is evicted once both have read
/// it, so by the end nearly every batch is gone.
#[test]
fn union_all_evicts_behind_both_scans() {
    let con = connect();
    let mut upstream = Upstream::new(id_batches(50, 1000));
    register(&con, "t", &mut upstream);
    assert_matches_reference(&con, UNION_ALL, 50_000);
    let (ingested, evicted, sealed) = stats(&con, "t");
    assert_eq!((ingested, sealed), (50, true));
    assert!(evicted >= 48, "evicted {evicted} of 50");
    assert_eq!(upstream.pulls(), 50);
}

/// A CTE referenced twice is materialized by DuckDB: one scan, one reader.
/// `NOT MATERIALIZED` gives two scans and two readers.  Both read the
/// upstream once.
#[test]
fn cte_referenced_twice() {
    for kind in ["", "NOT MATERIALIZED"] {
        let con = connect();
        let mut upstream = Upstream::new(id_batches(10, 1000));
        register(&con, "t", &mut upstream);
        let query = format!(
            "WITH c AS {kind} (SELECT * FROM {{t}}) \
             SELECT x.id, y.m FROM c x JOIN c y USING (id) ORDER BY x.id"
        );
        assert_matches_reference(&con, &query, 10_000);
        assert_eq!(upstream.pulls(), 10, "{kind}");
    }
}

/// A scan the optimizer prunes releases its reservation before the query
/// runs, so the surviving scan still evicts.
#[test]
fn pruned_scan_releases_its_reservation() {
    let con = connect();
    let mut upstream = Upstream::new(id_batches(10, 1000));
    register(&con, "t", &mut upstream);
    assert_matches_reference(
        &con,
        "SELECT id, m FROM (SELECT * FROM {t} UNION ALL SELECT * FROM {t} WHERE false) ORDER BY id",
        10_000,
    );
    let (_, evicted, _) = stats(&con, "t");
    assert!(evicted >= 9, "evicted {evicted} of 10");
}

#[test]
fn sealed_tee_cannot_serve_a_second_query() {
    let con = connect();
    let mut upstream = Upstream::new(id_batches(10, 1000));
    register(&con, "t", &mut upstream);
    con.query_row("SELECT count(*) FROM tee_scan('t')", [], |r| {
        r.get::<_, i64>(0)
    })
    .expect("first query");
    let err = query_error(&con, "SELECT count(*) FROM tee_scan('t')");
    assert!(err.contains("cannot read tee 't' from the start"), "{err}");
    assert!(err.contains("bounding := 'unbounded'"), "{err}");
}

#[test]
fn unbounded_tee_serves_any_number_of_queries() {
    let con = connect();
    let mut upstream = Upstream::new(id_batches(10, 1000));
    register_bounded(&con, "t", &mut upstream, "unbounded");
    for _ in 0..3 {
        assert_matches_reference(&con, UNION_ALL, 10_000);
    }
    assert_eq!(stats(&con, "t"), (10, 0, false));
    assert_eq!(upstream.pulls(), 10);
}

/// DuckDB re-binds a prepared statement on every execution, so each run
/// reserves fresh readers: fine for an unbounded tee, an error for a sealed
/// one once the first run has evicted.
#[test]
fn prepared_statement_executed_twice() {
    for (bounding, second_ok) in [("unbounded", true), ("sealed", false)] {
        let con = connect();
        let mut upstream = Upstream::new(id_batches(10, 1000));
        register_bounded(&con, "t", &mut upstream, bounding);
        let mut stmt = con
            .prepare("SELECT count(*) FROM tee_scan('t') a JOIN tee_scan('t') b USING (id)")
            .expect("prepare");
        let first: i64 = stmt.query_row([], |r| r.get(0)).expect("first execution");
        assert_eq!(first, 10_000, "{bounding}");
        let second = stmt.query_row([], |r| r.get::<_, i64>(0));
        assert_eq!(second.is_ok(), second_ok, "{bounding}: {second:?}");
    }
}

/// A recursive CTE re-initialises its scan once per iteration.  An
/// unbounded tee replays it; a sealed one has evicted by then and says so.
#[test]
fn recursive_cte_needs_an_unbounded_tee() {
    let query = "WITH RECURSIVE r(n) AS (SELECT 0 UNION ALL \
                   SELECT r.n + 1 FROM r JOIN tee_scan('t') p ON p.id = r.n WHERE r.n < 3) \
                 SELECT count(*) FROM r";
    let con = connect();
    let mut upstream = Upstream::new(id_batches(10, 1000));
    register_bounded(&con, "t", &mut upstream, "unbounded");
    let n: i64 = con.query_row(query, [], |r| r.get(0)).expect("unbounded");
    assert_eq!(n, 4);

    let con = connect();
    let mut upstream = Upstream::new(id_batches(10, 1000));
    register(&con, "t", &mut upstream);
    let err = query_error(&con, query);
    assert!(err.contains("recursive CTE"), "{err}");
}

/// `bounding` is checked at bind time, before the stream is taken.
#[test]
fn bad_bounding_is_rejected_and_leaves_the_stream() {
    let con = connect();
    for bounding in ["'counted:2'", "'nope'", "NULL"] {
        let mut upstream = Upstream::new(id_batches(1, 10));
        let sql = format!(
            "CALL tee_register('t', {}::UBIGINT, bounding := {bounding})",
            upstream.ptr()
        );
        let err = con.execute_batch(&sql).expect_err(&sql).to_string();
        assert!(
            err.contains("bounding must be 'sealed' or 'unbounded'"),
            "{err}"
        );
        assert!(
            upstream.stream.release.is_some(),
            "{bounding}: stream untouched"
        );
    }
}
