"""DuckDB self-join integration test for CachedDataset.

Validates the core value proposition: a single upstream RecordBatchReader
can be wrapped once in a CachedDataset, and two independent reader handles
can be registered as separate DuckDB virtual tables — each replaying the
same cached data without re-draining the source.
"""

import pyarrow as pa
import duckdb
import pytest

from multirecord import CachedDataset


# ── counting wrapper ──────────────────────────────────────────────────────────

class CountingReader:
    """Thin Python wrapper that counts how many batches the upstream has yielded.

    Implements ``__arrow_c_stream__`` so it can be passed directly to
    ``CachedDataset``.  Because PyArrow's ``RecordBatchReader.from_batches``
    accepts a lazy generator, the counter increments each time Rust asks for
    the next batch through the C Stream interface — giving an exact measure of
    how many times the original source was read.
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

        reader = pa.RecordBatchReader.from_batches(self._schema, _gen())
        return reader.__arrow_c_stream__(requested_schema)


# ── fixture ───────────────────────────────────────────────────────────────────

SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("name", pa.string()),
    ("manager_id", pa.int64()),  # null for top-level employees
])

BATCHES = [
    pa.record_batch({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"],
        "manager_id": [None, 1, 1],
    }, schema=SCHEMA),
    pa.record_batch({
        "id": [4, 5, 6],
        "name": ["Dave", "Eve", "Frank"],
        "manager_id": [2, 2, 3],
    }, schema=SCHEMA),
]

TOTAL_BATCHES = len(BATCHES)  # 2


@pytest.fixture()
def employee_dataset(tmp_path):
    """Yield a (CachedDataset, CountingReader) pair backed by the employee data."""
    source = CountingReader(BATCHES, SCHEMA)
    ds = CachedDataset(
        source,
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )
    return ds, source


# ── tests ─────────────────────────────────────────────────────────────────────

class TestDuckDBSelfJoin:
    def test_self_join_results(self, employee_dataset):
        """The self-join resolves each employee's manager name correctly."""
        ds, _ = employee_dataset

        con = duckdb.connect()
        con.register("employees", ds.reader())
        con.register("managers", ds.reader())

        rows = con.execute("""
            SELECT e.name AS employee, m.name AS manager
            FROM employees e
            JOIN managers m ON e.manager_id = m.id
            ORDER BY e.name
        """).fetchall()

        assert rows == [
            ("Bob",   "Alice"),
            ("Carol", "Alice"),
            ("Dave",  "Bob"),
            ("Eve",   "Bob"),
            ("Frank", "Carol"),
        ]

    def test_source_consumed_exactly_once(self, employee_dataset):
        """The upstream reader is drained only once regardless of reader count.

        Uses both the CountingReader wrapper (batch-level) and the dataset's
        built-in ``ingested_count`` property to verify this from two angles.
        """
        ds, source = employee_dataset

        con = duckdb.connect()
        con.register("employees", ds.reader())
        con.register("managers", ds.reader())

        con.execute("""
            SELECT e.name, m.name
            FROM employees e
            JOIN managers m ON e.manager_id = m.id
        """).fetchall()

        # The counting wrapper proves the upstream generator was iterated
        # exactly TOTAL_BATCHES times — not 2× for each reader.
        assert source.batches_read == TOTAL_BATCHES, (
            f"Expected upstream to be read {TOTAL_BATCHES} times, "
            f"got {source.batches_read}"
        )

        # The dataset's own ingestion counter independently confirms this.
        assert ds.ingested_count == TOTAL_BATCHES
        assert ds.upstream_exhausted is True

    def test_alice_has_no_manager_row(self, employee_dataset):
        """Alice (manager_id=NULL) is correctly absent from the join output."""
        ds, _ = employee_dataset

        con = duckdb.connect()
        con.register("employees", ds.reader())
        con.register("managers", ds.reader())

        employees_in_result = [
            row[0]
            for row in con.execute("""
                SELECT e.name
                FROM employees e
                JOIN managers m ON e.manager_id = m.id
            """).fetchall()
        ]

        assert "Alice" not in employees_in_result

    def test_all_non_root_employees_have_manager(self, employee_dataset):
        """Every employee with a non-null manager_id appears in the join."""
        ds, _ = employee_dataset

        con = duckdb.connect()
        con.register("employees", ds.reader())
        con.register("managers", ds.reader())

        result_names = {
            row[0]
            for row in con.execute("""
                SELECT e.name
                FROM employees e
                JOIN managers m ON e.manager_id = m.id
            """).fetchall()
        }

        assert result_names == {"Bob", "Carol", "Dave", "Eve", "Frank"}


class TestBareReaderFailure:
    """Documents the failure mode that CachedDataset is designed to prevent.

    A bare ``pa.RecordBatchReader`` is a single-use stream: once DuckDB scans
    it for the first virtual table, it is exhausted.  The second virtual table
    therefore sees zero rows, and the join silently returns no results — a
    correctness bug that is easy to miss because no exception is raised.
    """

    def test_bare_reader_self_join_returns_empty(self):
        """Registering the same RecordBatchReader twice yields 0 join rows.

        DuckDB scans ``employees`` first, draining the stream entirely.
        When it then scans ``managers``, the reader is already exhausted so
        the join finds no matching rows on either side.
        """
        reader = pa.RecordBatchReader.from_batches(SCHEMA, iter(BATCHES))

        con = duckdb.connect()
        con.register("employees", reader)
        con.register("managers", reader)

        rows = con.execute("""
            SELECT e.name AS employee, m.name AS manager
            FROM employees e
            JOIN managers m ON e.manager_id = m.id
            ORDER BY e.name
        """).fetchall()

        # Contrast with the 5-row result produced by CachedDataset.
        assert rows == [], (
            "Expected the bare-reader join to return no rows because the "
            "stream is exhausted after the first scan, but got: {rows!r}"
        )
