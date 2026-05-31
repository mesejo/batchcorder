"""Tests for the max_readers parameter and low-water-mark eviction."""

import gc

import pyarrow as pa
import pytest

from batchcorder import StreamCache


def _make_table(n_batches: int = 4, rows_per_batch: int = 3) -> pa.Table:
    n = n_batches * rows_per_batch
    return pa.table(
        {
            "id": list(range(n)),
            "value": [float(i) * 1.5 for i in range(n)],
        }
    )


def _memory_cache(table=None, batch_size=3, max_readers=None):
    if table is None:
        table = _make_table()
    return StreamCache(
        table.to_reader(max_chunksize=batch_size),
        max_readers=max_readers,
    )


def _disk_cache(tmp_path, table=None, batch_size=3, max_readers=None):
    if table is None:
        table = _make_table()
    return StreamCache(
        table.to_reader(max_chunksize=batch_size),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
        max_readers=max_readers,
    )


# ── max_readers=None preserves existing behavior ────────────────────────────


def test_none_allows_unlimited_readers():
    ds = _memory_cache(max_readers=None)
    readers = [ds.reader() for _ in range(10)]
    assert len(readers) == 10


def test_none_retains_all_batches():
    ds = _memory_cache(max_readers=None)
    r1 = ds.reader()
    batches_1 = list(r1)
    r2 = ds.reader()
    batches_2 = list(r2)
    assert len(batches_1) == len(batches_2) == 4


# ── max_readers enforcement ─────────────────────────────────────────────────


def test_max_readers_zero_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        _memory_cache(max_readers=0)


def test_max_readers_enforced():
    ds = _memory_cache(max_readers=2)
    ds.reader()
    ds.reader()
    with pytest.raises(ValueError, match="Maximum number of readers"):
        ds.reader()


def test_max_readers_enforced_via_iter():
    ds = _memory_cache(max_readers=1)
    _ = iter(ds)
    with pytest.raises(ValueError, match="Maximum number of readers"):
        ds.reader()


def test_max_readers_enforced_via_arrow_c_stream():
    ds = _memory_cache(max_readers=1)
    _ = ds.__arrow_c_stream__()
    with pytest.raises(ValueError, match="Maximum number of readers"):
        ds.reader()


def test_max_readers_exhausted_blocks_from_start_true():
    ds = _memory_cache(max_readers=1)
    r = ds.reader()
    list(r)
    with pytest.raises(ValueError, match="Maximum number of readers"):
        ds.reader(from_start=True)


def test_max_readers_exhausted_blocks_from_start_false():
    ds = _memory_cache(max_readers=1)
    r = ds.reader()
    list(r)
    with pytest.raises(ValueError, match="Maximum number of readers"):
        ds.reader(from_start=False)


# ── single reader reads correct data despite eviction ────────────────────────


def test_single_reader_reads_complete_data():
    table = _make_table(n_batches=4)
    ds = _memory_cache(table=table, max_readers=1)
    r = ds.reader()
    result = pa.RecordBatchReader.from_stream(r).read_all()
    assert result.equals(table)


def test_single_reader_iterates_all_batches():
    ds = _memory_cache(max_readers=1)
    r = ds.reader()
    batches = list(r)
    assert len(batches) == 4


# ── two parallel readers with eviction ───────────────────────────────────────


def test_two_readers_both_get_correct_data():
    table = _make_table(n_batches=4)
    ds = _memory_cache(table=table, max_readers=2)
    r1 = ds.reader()
    r2 = ds.reader()
    result1 = pa.RecordBatchReader.from_stream(r1).read_all()
    result2 = pa.RecordBatchReader.from_stream(r2).read_all()
    assert result1.equals(table)
    assert result2.equals(table)


def test_two_readers_interleaved_get_same_batches():
    table = _make_table(n_batches=4)
    ds = _memory_cache(table=table, max_readers=2)
    r1 = ds.reader()
    r2 = ds.reader()
    for _ in range(4):
        b1 = next(r1)
        b2 = next(r2)
        assert b1 == b2


def test_eviction_happens_when_both_advance():
    table = _make_table(n_batches=4)
    ds = _memory_cache(table=table, max_readers=2)
    r1 = ds.reader()
    r2 = ds.reader()
    next(r1)
    next(r1)
    next(r2)
    next(r2)
    last1 = next(r1)
    last2 = next(r2)
    assert last1 == last2


# ── dropped reader unblocks eviction ─────────────────────────────────────────


def test_dropped_reader_allows_eviction_to_advance():
    table = _make_table(n_batches=4)
    ds = _memory_cache(table=table, max_readers=2)
    r1 = ds.reader()
    r2 = ds.reader()
    next(r1)
    next(r1)
    next(r1)
    next(r1)
    del r2
    gc.collect()
    # Trigger eviction by attempting to read again (StopIteration is fine)
    with pytest.raises(StopIteration):
        next(r1)


# ── CastingStreamCache respects max_readers ──────────────────────────────────


def test_cast_respects_max_readers():
    table = _make_table(n_batches=2)
    ds = _memory_cache(table=table, max_readers=1)
    casted = ds.cast(ds.schema)
    _ = pa.RecordBatchReader.from_stream(casted).read_all()
    with pytest.raises(ValueError, match="Maximum number of readers"):
        pa.RecordBatchReader.from_stream(casted).read_all()


def test_cast_reads_correct_data():
    table = _make_table(n_batches=2)
    ds = _memory_cache(table=table, max_readers=1)
    casted = ds.cast(ds.schema)
    result = pa.RecordBatchReader.from_stream(casted).read_all()
    assert result.equals(table)


# ── disk tier ────────────────────────────────────────────────────────────────


def test_disk_single_reader_correct_data(tmp_path):
    table = _make_table(n_batches=4)
    ds = _disk_cache(tmp_path, table=table, max_readers=1)
    r = ds.reader()
    result = pa.RecordBatchReader.from_stream(r).read_all()
    assert result.equals(table)


def test_disk_two_readers_correct_data(tmp_path):
    table = _make_table(n_batches=4)
    ds = _disk_cache(tmp_path, table=table, max_readers=2)
    r1 = ds.reader()
    r2 = ds.reader()
    result1 = pa.RecordBatchReader.from_stream(r1).read_all()
    result2 = pa.RecordBatchReader.from_stream(r2).read_all()
    assert result1.equals(table)
    assert result2.equals(table)


def test_disk_max_readers_enforced(tmp_path):
    ds = _disk_cache(tmp_path, max_readers=2)
    ds.reader()
    ds.reader()
    with pytest.raises(ValueError, match="Maximum number of readers"):
        ds.reader()
