"""DuckDB integration tests for CachedDataset.

Each test asserts both sides of the same scenario:
  - CachedDataset behaves correctly (the passing case)
  - A bare pa.RecordBatchReader fails silently (the motivating failure)
"""

import pyarrow as pa
import duckdb

from multirecord import CachedDataset


# ── shared data ───────────────────────────────────────────────────────────────

EMPLOYEE_SCHEMA = pa.schema([
    ("id",         pa.int64()),
    ("name",       pa.string()),
    ("manager_id", pa.int64()),  # null for root employees
])

EMPLOYEE_BATCHES = [
    pa.record_batch({"id": [1, 2, 3], "name": ["Alice", "Bob",   "Carol"],
                     "manager_id": [None, 1, 1]}, schema=EMPLOYEE_SCHEMA),
    pa.record_batch({"id": [4, 5, 6], "name": ["Dave",  "Eve",   "Frank"],
                     "manager_id": [2, 2, 3]},    schema=EMPLOYEE_SCHEMA),
]

LARGE_SCHEMA = pa.schema([("id", pa.int64()), ("value", pa.float64())])
LARGE_ROWS_PER_BATCH = 100
LARGE_TOTAL_BATCHES = 10
LARGE_TOTAL_ROWS = LARGE_ROWS_PER_BATCH * LARGE_TOTAL_BATCHES  # 1 000


def _large_batches() -> list[pa.RecordBatch]:
    return [
        pa.record_batch({
            "id":    list(range(i * LARGE_ROWS_PER_BATCH, (i + 1) * LARGE_ROWS_PER_BATCH)),
            "value": [float(j) for j in range(i * LARGE_ROWS_PER_BATCH, (i + 1) * LARGE_ROWS_PER_BATCH)],
        }, schema=LARGE_SCHEMA)
        for i in range(LARGE_TOTAL_BATCHES)
    ]


class CountingReader:
    """Wraps a batch list and counts how many batches the upstream has yielded.

    Implements ``__arrow_c_stream__`` via a lazy generator so the counter
    increments at the C-stream level as each batch is actually consumed.
    """

    def __init__(self, batches: list[pa.RecordBatch], schema: pa.Schema) -> None:
        self._batches = batches
        self._schema = schema
        self.batches_read = 0

    def __arrow_c_stream__(self, requested_schema=None):
        def _gen():
            for batch in self._batches:
                self.batches_read += 1
                yield batch
        return pa.RecordBatchReader.from_batches(self._schema, _gen()).__arrow_c_stream__(requested_schema)


# ── self-join ─────────────────────────────────────────────────────────────────

def test_self_join(tmp_path):
    """CachedDataset: self-join returns the correct manager mapping.
    Bare reader: self-join silently returns zero rows.

    DuckDB calls __arrow_c_stream__ once per side of the join.  CachedDataset
    replays from the cache on each call; a bare reader is exhausted after the
    first scan, so the second side sees nothing and the join produces no rows.
    """
    source = CountingReader(EMPLOYEE_BATCHES, EMPLOYEE_SCHEMA)
    ds = CachedDataset(source, 16 * 1024 * 1024, str(tmp_path), 64 * 1024 * 1024)

    con = duckdb.connect()
    con.register("employees", ds)

    rows = con.execute("""
        SELECT e.name AS employee, m.name AS manager
        FROM employees e
        JOIN employees m ON e.manager_id = m.id
        ORDER BY e.name
    """).fetchall()

    # CachedDataset: correct result
    assert rows == [
        ("Bob",   "Alice"),
        ("Carol", "Alice"),
        ("Dave",  "Bob"),
        ("Eve",   "Bob"),
        ("Frank", "Carol"),
    ]
    assert "Alice" not in {r[0] for r in rows}  # root employee excluded

    # Source was ingested exactly once regardless of how many times DuckDB scanned
    assert source.batches_read == len(EMPLOYEE_BATCHES)
    assert ds.ingested_count == len(EMPLOYEE_BATCHES)
    assert ds.upstream_exhausted is True

    # Bare reader: same query returns empty because the stream is single-use
    bare = pa.RecordBatchReader.from_batches(EMPLOYEE_SCHEMA, iter(EMPLOYEE_BATCHES))
    con_bare = duckdb.connect()
    con_bare.register("employees", bare)

    bare_rows = con_bare.execute("""
        SELECT e.name AS employee, m.name AS manager
        FROM employees e
        JOIN employees m ON e.manager_id = m.id
        ORDER BY e.name
    """).fetchall()

    assert bare_rows == []


# ── partial consumption under LIMIT ──────────────────────────────────────────

def test_limit_does_not_exhaust_upstream(tmp_path):
    """CachedDataset: a LIMIT query leaves the upstream open for further reads.
    Bare reader: the same LIMIT query destructively advances the stream,
    so a subsequent full-table query returns fewer than LARGE_TOTAL_ROWS.
    """
    batches = _large_batches()

    # ── CachedDataset ────────────────────────────────────────────────────────
    ds = CachedDataset(
        pa.RecordBatchReader.from_batches(LARGE_SCHEMA, iter(batches)),
        memory_capacity=64 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=256 * 1024 * 1024,
    )
    con = duckdb.connect()
    con.register("data", ds)

    limited = con.execute("SELECT * FROM data LIMIT 50").fetchall()
    assert len(limited) == 50
    # Upstream not exhausted: DuckDB read ahead a few batches but stopped early
    assert ds.ingested_count < LARGE_TOTAL_BATCHES
    assert ds.upstream_exhausted is False

    # A follow-up full scan still gets every row (cache + remaining upstream)
    total = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    assert total == LARGE_TOTAL_ROWS
    assert ds.upstream_exhausted is True

    # ── Bare reader ───────────────────────────────────────────────────────────
    bare = pa.RecordBatchReader.from_batches(LARGE_SCHEMA, iter(batches))
    con_bare = duckdb.connect()
    con_bare.register("data", bare)

    bare_limited = con_bare.execute("SELECT * FROM data LIMIT 50").fetchall()
    assert len(bare_limited) == 50

    # After the LIMIT query the stream is partially spent; the second query
    # cannot recover the discarded batches and sees fewer than LARGE_TOTAL_ROWS.
    bare_total = con_bare.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    assert bare_total < LARGE_TOTAL_ROWS
