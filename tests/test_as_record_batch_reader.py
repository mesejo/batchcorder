"""Tests verifying that CachedDataset and CachedDatasetReader behave as
well-behaved Arrow stream sources — compatible with PyArrow, arro3, IPC
serialisation, and the full Arrow PyCapsule protocol.

Inspired by /home/daniel/PycharmProjects/arro3/tests/core/.
"""

import io

import arro3.core as ac
import pyarrow as pa
import pyarrow.ipc
from batchcorder import CachedDataset

# ── helpers ───────────────────────────────────────────────────────────────────


def _ds(tmp_path, table, batch_size=3):
    return CachedDataset(
        table.to_reader(max_chunksize=batch_size),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )


TABLE = pa.table(
    {
        "id": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
        "label": pa.array(["a", "b", "c", "d", "e", "f"], type=pa.utf8()),
    }
)


# ── pa.table() constructor ────────────────────────────────────────────────────


def test_pa_table_from_dataset(tmp_path):
    """pa.table(ds) should materialise all rows correctly."""
    assert pa.table(_ds(tmp_path, TABLE)).equals(TABLE)


def test_pa_table_from_reader(tmp_path):
    """pa.table(ds.reader()) should materialise all rows correctly."""
    assert pa.table(_ds(tmp_path, TABLE).reader()).equals(TABLE)


# ── pa.RecordBatchReader interop ──────────────────────────────────────────────


def test_pa_record_batch_reader_from_stream_dataset(tmp_path):
    """`pa.RecordBatchReader.from_stream` accepts CachedDataset directly."""
    result = pa.RecordBatchReader.from_stream(_ds(tmp_path, TABLE)).read_all()
    assert result.equals(TABLE)


def test_pa_record_batch_reader_from_stream_reader(tmp_path):
    """`pa.RecordBatchReader.from_stream` accepts CachedDatasetReader."""
    result = pa.RecordBatchReader.from_stream(_ds(tmp_path, TABLE).reader()).read_all()
    assert result.equals(TABLE)


def test_batch_iteration_yields_correct_batches(tmp_path):
    """Iterating the PyArrow reader batch-by-batch preserves per-batch contents."""
    ds = _ds(tmp_path, TABLE, batch_size=2)
    pa_reader = pa.RecordBatchReader.from_stream(ds)

    batches = list(pa_reader)
    assert sum(b.num_rows for b in batches) == TABLE.num_rows
    assert pa.Table.from_batches(batches).equals(TABLE)


# ── schema metadata preservation ─────────────────────────────────────────────


def test_schema_metadata_preserved_through_cache(tmp_path):
    """Schema metadata survives the IPC serialisation round-trip inside the cache."""
    metadata = {b"author": b"test", b"version": b"1"}
    table = TABLE.replace_schema_metadata(metadata)

    result = pa.RecordBatchReader.from_stream(_ds(tmp_path, table)).read_all()
    assert result.schema.metadata == metadata


def test_schema_metadata_preserved_through_reader(tmp_path):
    """Metadata is intact when consuming via a CachedDatasetReader handle."""
    metadata = {b"hello": b"world"}
    table = TABLE.replace_schema_metadata(metadata)

    result = pa.RecordBatchReader.from_stream(_ds(tmp_path, table).reader()).read_all()
    assert result.schema.metadata == metadata


# ── requested_schema (C stream protocol type casting) ────────────────────────


def test_requested_schema_casts_column_type_on_dataset(tmp_path):
    """Passing a requested_schema to __arrow_c_stream__ triggers a cast.

    This exercises the Arrow C Stream protocol's schema negotiation: the
    consumer asks for large_utf8 and the producer (pyo3-arrow) casts
    transparently.
    """
    ds = _ds(tmp_path, TABLE)
    requested = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("label", pa.large_utf8()),
        ]
    )
    result = pa.RecordBatchReader.from_stream(ds, schema=requested).read_all()
    assert result.schema.field("label").type == pa.large_utf8()
    assert result.column("label").to_pylist() == TABLE.column("label").to_pylist()


def test_requested_schema_casts_column_type_on_reader(tmp_path):
    """Same requested_schema negotiation works on a CachedDatasetReader."""
    r = _ds(tmp_path, TABLE).reader()
    requested = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("label", pa.large_utf8()),
        ]
    )
    result = pa.RecordBatchReader.from_stream(r, schema=requested).read_all()
    assert result.schema.field("label").type == pa.large_utf8()


# ── IPC round-trip ────────────────────────────────────────────────────────────


def test_ipc_stream_round_trip_dataset(tmp_path):
    """CachedDataset can be piped into an IPC stream writer and read back."""
    pa_reader = pa.RecordBatchReader.from_stream(_ds(tmp_path, TABLE))
    buf = io.BytesIO()
    with pa.ipc.new_stream(buf, pa_reader.schema) as writer:
        for batch in pa_reader:
            writer.write_batch(batch)
    buf.seek(0)
    assert pa.ipc.open_stream(buf).read_all().equals(TABLE)


def test_ipc_stream_round_trip_reader(tmp_path):
    """CachedDatasetReader can be piped into an IPC stream writer and read back."""
    pa_reader = pa.RecordBatchReader.from_stream(_ds(tmp_path, TABLE).reader())
    buf = io.BytesIO()
    with pa.ipc.new_stream(buf, pa_reader.schema) as writer:
        for batch in pa_reader:
            writer.write_batch(batch)
    buf.seek(0)
    assert pa.ipc.open_stream(buf).read_all().equals(TABLE)


# ── arro3 interop ─────────────────────────────────────────────────────────────


def test_arro3_from_stream_dataset(tmp_path):
    """`arro3.core.RecordBatchReader.from_stream` accepts CachedDataset."""
    arro3_reader = ac.RecordBatchReader.from_stream(_ds(tmp_path, TABLE))
    assert pa.table(arro3_reader).equals(TABLE)


def test_arro3_from_stream_reader(tmp_path):
    """`arro3.core.RecordBatchReader.from_stream` accepts CachedDatasetReader."""
    arro3_reader = ac.RecordBatchReader.from_stream(_ds(tmp_path, TABLE).reader())
    assert pa.table(arro3_reader).equals(TABLE)


def test_arro3_schema_metadata_preserved(tmp_path):
    """arro3 preserves schema metadata when consuming a CachedDataset."""
    metadata = {b"hello": b"world"}
    table = TABLE.replace_schema_metadata(metadata)

    arro3_reader = ac.RecordBatchReader.from_stream(_ds(tmp_path, table))
    assert (
        pa.RecordBatchReader.from_stream(arro3_reader).read_all().schema.metadata
        == metadata
    )
