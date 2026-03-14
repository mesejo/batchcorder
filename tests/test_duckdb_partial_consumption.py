"""Tests for partial stream consumption under a DuckDB LIMIT clause.

When a query carries a LIMIT that is smaller than the total row count, DuckDB
stops pulling from the Arrow stream once it has collected enough rows.  The
tests in this file verify two complementary behaviours:

* ``TestCachedDatasetPartialRead`` — with CachedDataset the upstream is only
  ingested as far as DuckDB needed; the remainder stays available for later
  queries that ingested from the cache plus, if necessary, further upstream
  reads.

* ``TestBareReaderPartialRead`` — a bare ``pa.RecordBatchReader`` is consumed
  destructively: a LIMIT query advances (and discards) part of the stream, so
  a follow-up full-table query misses the rows that were silently skipped.
"""

import pyarrow as pa
import duckdb
import pytest

from multirecord import CachedDataset


# ── shared data ───────────────────────────────────────────────────────────────

# 10 batches × 100 rows = 1 000 rows total.
# A LIMIT 50 query will cause DuckDB to stop after a handful of batches,
# leaving the majority of the stream unread.
ROWS_PER_BATCH = 100
TOTAL_BATCHES = 10
TOTAL_ROWS = ROWS_PER_BATCH * TOTAL_BATCHES  # 1 000

SCHEMA = pa.schema([("id", pa.int64()), ("value", pa.float64())])


def _make_batches() -> list[pa.RecordBatch]:
    return [
        pa.record_batch(
            {
                "id": list(range(i * ROWS_PER_BATCH, (i + 1) * ROWS_PER_BATCH)),
                "value": [float(j) for j in range(i * ROWS_PER_BATCH, (i + 1) * ROWS_PER_BATCH)],
            },
            schema=SCHEMA,
        )
        for i in range(TOTAL_BATCHES)
    ]


# ── CachedDataset ─────────────────────────────────────────────────────────────

class TestCachedDatasetPartialRead:
    def test_limit_does_not_exhaust_upstream(self, tmp_path):
        """A LIMIT query reads only as many batches as needed; the upstream is
        left open so subsequent queries can lazily ingest the remainder.
        """
        batches = _make_batches()
        ds = CachedDataset(
            pa.RecordBatchReader.from_batches(SCHEMA, iter(batches)),
            memory_capacity=64 * 1024 * 1024,
            disk_path=str(tmp_path),
            disk_capacity=256 * 1024 * 1024,
        )

        con = duckdb.connect()
        con.register("data", ds)

        rows = con.execute("SELECT * FROM data LIMIT 50").fetchall()

        assert len(rows) == 50
        # DuckDB stopped mid-stream: fewer than all batches were ingested …
        assert ds.ingested_count < TOTAL_BATCHES
        # … and the upstream reader is still open.
        assert ds.upstream_exhausted is False

    def test_full_query_after_partial_read(self, tmp_path):
        """After a partial LIMIT read, a subsequent full query still returns
        all rows — the cache serves what was already ingested and the upstream
        is drained lazily for the rest.
        """
        batches = _make_batches()
        ds = CachedDataset(
            pa.RecordBatchReader.from_batches(SCHEMA, iter(batches)),
            memory_capacity=64 * 1024 * 1024,
            disk_path=str(tmp_path),
            disk_capacity=256 * 1024 * 1024,
        )

        con = duckdb.connect()
        con.register("data", ds)

        # Partial read first …
        con.execute("SELECT * FROM data LIMIT 50").fetchall()
        assert ds.upstream_exhausted is False

        # … then full scan: must still return every row.
        all_rows = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
        assert all_rows == TOTAL_ROWS
        assert ds.upstream_exhausted is True


# ── bare reader ───────────────────────────────────────────────────────────────

class TestBareReaderPartialRead:
    def test_limit_partially_advances_bare_reader(self):
        """A LIMIT query on a bare RecordBatchReader advances the stream by
        however many batches DuckDB pulled internally, discarding those rows.
        A subsequent full-table query therefore does not see the rows that
        DuckDB already consumed and threw away.
        """
        batches = _make_batches()
        reader = pa.RecordBatchReader.from_batches(SCHEMA, iter(batches))

        con = duckdb.connect()
        con.register("data", reader)

        # First query: returns exactly 50 rows as requested.
        limited_rows = con.execute("SELECT * FROM data LIMIT 50").fetchall()
        assert len(limited_rows) == 50

        # Second query: the stream has advanced past the batches DuckDB
        # already read; the returned count is less than the full 1 000 rows.
        remaining_rows = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
        assert remaining_rows < TOTAL_ROWS, (
            f"Expected the second query to see fewer than {TOTAL_ROWS} rows "
            f"because the bare reader was partially consumed, "
            f"but got {remaining_rows}"
        )
