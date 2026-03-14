"""Tests for CachedDataset and CachedDatasetReader."""

import tempfile
import threading

import pyarrow as pa
import pytest

from multirecord import CachedDataset, CachedDatasetReader


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_table(n_batches: int = 4, rows_per_batch: int = 3) -> pa.Table:
    """Create a small multi-batch PyArrow table."""
    return pa.table({
        "id": list(range(n_batches * rows_per_batch)),
        "value": [float(i) * 1.5 for i in range(n_batches * rows_per_batch)],
        "label": [f"row_{i}" for i in range(n_batches * rows_per_batch)],
    })


def _make_reader(table: pa.Table, batch_size: int = 3) -> pa.RecordBatchReader:
    return table.to_reader(max_chunksize=batch_size)


def _dataset(tmp_path, table=None, batch_size=3, mem_mb=16, disk_mb=64):
    if table is None:
        table = _make_table()
    reader = _make_reader(table, batch_size)
    return CachedDataset(
        reader,
        memory_capacity=mem_mb * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=disk_mb * 1024 * 1024,
    )


# ── construction ─────────────────────────────────────────────────────────────

class TestConstruction:
    def test_basic_construction(self, tmp_path):
        ds = _dataset(tmp_path)
        assert ds is not None

    def test_accepts_pyarrow_table_directly(self, tmp_path):
        table = _make_table()
        ds = CachedDataset(
            table,  # pa.Table also implements __arrow_c_stream__
            memory_capacity=16 * 1024 * 1024,
            disk_path=str(tmp_path),
            disk_capacity=64 * 1024 * 1024,
        )
        assert ds is not None

    def test_ingested_count_starts_at_zero(self, tmp_path):
        ds = _dataset(tmp_path)
        assert ds.ingested_count == 0

    def test_upstream_not_exhausted_initially(self, tmp_path):
        ds = _dataset(tmp_path)
        assert ds.upstream_exhausted is False


# ── schema ───────────────────────────────────────────────────────────────────

def _schema_from_capsule(capsule) -> pa.Schema:
    """Import a pa.Schema from an Arrow C Schema PyCapsule."""
    # PyArrow ≥14 accepts objects implementing __arrow_c_schema__ via pa.schema().
    # We wrap the capsule in a thin shim so pa.schema() can consume it.
    class _Wrap:
        def __arrow_c_schema__(self):
            return capsule
    return pa.schema(_Wrap())


class TestSchema:
    def test_schema_via_c_schema_capsule(self, tmp_path):
        table = _make_table()
        ds = _dataset(tmp_path, table)
        schema = _schema_from_capsule(ds.__arrow_c_schema__())
        assert schema == table.schema

    def test_schema_property_matches_source(self, tmp_path):
        table = _make_table()
        ds = _dataset(tmp_path, table)
        # .schema returns an arro3.core.Schema which implements __arrow_c_schema__
        schema = pa.schema(ds.schema)
        assert schema == table.schema

    def test_reader_schema_matches_dataset(self, tmp_path):
        table = _make_table()
        ds = _dataset(tmp_path, table)
        r = ds.reader()
        schema = pa.schema(r.schema)
        assert schema == table.schema


# ── round-trip data integrity ─────────────────────────────────────────────────

class TestDataIntegrity:
    def test_reader_returns_all_rows(self, tmp_path):
        table = _make_table(n_batches=4, rows_per_batch=3)
        ds = _dataset(tmp_path, table, batch_size=3)
        r = ds.reader()
        result = pa.RecordBatchReader.from_stream(r).read_all()
        assert result.num_rows == table.num_rows

    def test_reader_data_matches_source(self, tmp_path):
        table = _make_table(n_batches=4, rows_per_batch=3)
        ds = _dataset(tmp_path, table, batch_size=3)
        r = ds.reader()
        result = pa.RecordBatchReader.from_stream(r).read_all()
        assert result.equals(table)

    def test_dataset_c_stream_data_matches_source(self, tmp_path):
        table = _make_table(n_batches=4, rows_per_batch=3)
        ds = _dataset(tmp_path, table, batch_size=3)
        result = pa.RecordBatchReader.from_stream(ds).read_all()
        assert result.equals(table)

    def test_single_batch(self, tmp_path):
        table = pa.table({"x": [1, 2, 3]})
        ds = _dataset(tmp_path, table, batch_size=100)
        result = pa.RecordBatchReader.from_stream(ds).read_all()
        assert result.equals(table)

    def test_many_small_batches(self, tmp_path):
        table = _make_table(n_batches=20, rows_per_batch=1)
        ds = _dataset(tmp_path, table, batch_size=1)
        result = pa.RecordBatchReader.from_stream(ds).read_all()
        assert result.equals(table)

    def test_empty_table(self, tmp_path):
        table = pa.table({"x": pa.array([], type=pa.int32())})
        ds = _dataset(tmp_path, table, batch_size=10)
        result = pa.RecordBatchReader.from_stream(ds).read_all()
        assert result.num_rows == 0
        assert result.schema.equals(table.schema)

    def test_various_dtypes(self, tmp_path):
        table = pa.table({
            "i32": pa.array([1, 2, 3], type=pa.int32()),
            "f64": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            "utf8": pa.array(["a", "b", "c"], type=pa.string()),
            "bool": pa.array([True, False, True], type=pa.bool_()),
            "bin": pa.array([b"x", b"y", b"z"], type=pa.binary()),
        })
        ds = _dataset(tmp_path, table)
        result = pa.RecordBatchReader.from_stream(ds).read_all()
        assert result.equals(table)

    def test_nullable_columns(self, tmp_path):
        table = pa.table({
            "x": pa.array([1, None, 3], type=pa.int64()),
            "y": pa.array(["a", None, "c"], type=pa.string()),
        })
        ds = _dataset(tmp_path, table, batch_size=2)
        result = pa.RecordBatchReader.from_stream(ds).read_all()
        assert result.equals(table)


# ── multiple readers ──────────────────────────────────────────────────────────

class TestMultipleReaders:
    def test_two_readers_same_data(self, tmp_path):
        table = _make_table(n_batches=4, rows_per_batch=3)
        ds = _dataset(tmp_path, table, batch_size=3)
        r1 = ds.reader()
        r2 = ds.reader()
        res1 = pa.RecordBatchReader.from_stream(r1).read_all()
        res2 = pa.RecordBatchReader.from_stream(r2).read_all()
        assert res1.equals(table)
        assert res2.equals(table)

    def test_multiple_readers_independent(self, tmp_path):
        table = _make_table(n_batches=6, rows_per_batch=2)
        ds = _dataset(tmp_path, table, batch_size=2)
        readers = [ds.reader() for _ in range(5)]
        results = [pa.RecordBatchReader.from_stream(r).read_all() for r in readers]
        for res in results:
            assert res.equals(table)

    def test_c_stream_creates_independent_readers(self, tmp_path):
        table = _make_table(n_batches=4, rows_per_batch=3)
        ds = _dataset(tmp_path, table, batch_size=3)
        # Each __arrow_c_stream__ call yields a fresh replay from batch 0
        res1 = pa.RecordBatchReader.from_stream(ds).read_all()
        res2 = pa.RecordBatchReader.from_stream(ds).read_all()
        assert res1.equals(table)
        assert res2.equals(table)


# ── reader lifecycle ──────────────────────────────────────────────────────────

class TestReaderLifecycle:
    def test_reader_closed_after_c_stream_export(self, tmp_path):
        ds = _dataset(tmp_path)
        r = ds.reader()
        assert r.closed is False
        pa.RecordBatchReader.from_stream(r).read_all()  # consumes via __arrow_c_stream__
        assert r.closed is True

    def test_reader_second_c_stream_raises(self, tmp_path):
        ds = _dataset(tmp_path)
        r = ds.reader()
        pa.RecordBatchReader.from_stream(r).read_all()
        with pytest.raises(Exception, match="consumed|closed"):
            r.__arrow_c_stream__()

    def test_reader_c_schema_raises_after_consumed(self, tmp_path):
        ds = _dataset(tmp_path)
        r = ds.reader()
        pa.RecordBatchReader.from_stream(r).read_all()
        with pytest.raises(Exception, match="consumed|closed"):
            r.__arrow_c_schema__()


# ── frontier reader ───────────────────────────────────────────────────────────

class TestFrontierReader:
    def test_from_start_false_after_ingest_all(self, tmp_path):
        table = _make_table(n_batches=4, rows_per_batch=3)
        ds = _dataset(tmp_path, table, batch_size=3)
        ds.ingest_all()
        # A reader from the frontier should yield nothing (all batches already past)
        r = ds.reader(from_start=False)
        result = pa.RecordBatchReader.from_stream(r).read_all()
        assert result.num_rows == 0

    def test_from_start_true_after_ingest_all(self, tmp_path):
        table = _make_table(n_batches=4, rows_per_batch=3)
        ds = _dataset(tmp_path, table, batch_size=3)
        ds.ingest_all()
        r = ds.reader(from_start=True)
        result = pa.RecordBatchReader.from_stream(r).read_all()
        assert result.equals(table)

    def test_frontier_reader_mid_stream(self, tmp_path):
        table = _make_table(n_batches=6, rows_per_batch=2)
        ds = _dataset(tmp_path, table, batch_size=2)
        # Consume first 3 batches (= 6 rows) via a reader, advancing the frontier
        r_first = ds.reader()
        half_reader = pa.RecordBatchReader.from_stream(r_first)
        first_half_batches = [half_reader.read_next_batch() for _ in range(3)]
        assert ds.ingested_count == 3
        # A frontier reader now starts at batch 3
        r_second = ds.reader(from_start=False)
        result = pa.RecordBatchReader.from_stream(r_second).read_all()
        assert result.num_rows == 6  # remaining 3 batches × 2 rows


# ── ingestion helpers ─────────────────────────────────────────────────────────

class TestIngestion:
    def test_ingest_all_returns_batch_count(self, tmp_path):
        table = _make_table(n_batches=5, rows_per_batch=2)
        ds = _dataset(tmp_path, table, batch_size=2)
        count = ds.ingest_all()
        assert count == 5

    def test_ingested_count_increments_lazily(self, tmp_path):
        table = _make_table(n_batches=4, rows_per_batch=3)
        ds = _dataset(tmp_path, table, batch_size=3)
        assert ds.ingested_count == 0
        r = ds.reader()
        batch_reader = pa.RecordBatchReader.from_stream(r)
        batch_reader.read_next_batch()
        assert ds.ingested_count >= 1

    def test_upstream_exhausted_after_ingest_all(self, tmp_path):
        ds = _dataset(tmp_path)
        assert ds.upstream_exhausted is False
        ds.ingest_all()
        assert ds.upstream_exhausted is True

    def test_ingest_all_idempotent(self, tmp_path):
        table = _make_table(n_batches=4, rows_per_batch=3)
        ds = _dataset(tmp_path, table, batch_size=3)
        count1 = ds.ingest_all()
        count2 = ds.ingest_all()
        assert count1 == count2 == 4


# ── threading ─────────────────────────────────────────────────────────────────

class TestThreading:
    def test_concurrent_readers(self, tmp_path):
        table = _make_table(n_batches=8, rows_per_batch=5)
        ds = _dataset(tmp_path, table, batch_size=5)
        results = [None] * 4
        errors = []

        def read(i):
            try:
                r = ds.reader()
                results[i] = pa.RecordBatchReader.from_stream(r).read_all()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        for res in results:
            assert res is not None
            assert res.equals(table)
