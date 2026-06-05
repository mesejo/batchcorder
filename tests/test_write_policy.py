"""Tests for the StreamCache write_policy parameter (Gap 5).

Covers both WriteOnInsertion (default) and WriteOnEviction:
  - data round-trips correctly under either policy
  - WriteOnEviction keeps disk empty when the hot layer fits the whole stream
  - disk_capacity is still enforced at eviction time
  - multiple independent readers see consistent data
  - invalid / memory-only policy handling
"""

import threading

import pyarrow as pa
import pytest

from batchcorder import StreamCache


def _make_source(n_rows: int, chunk: int) -> tuple[pa.RecordBatchReader, pa.Table]:
    t = pa.table({"id": list(range(n_rows)), "v": [str(i) * 10 for i in range(n_rows)]})
    return t.to_reader(max_chunksize=chunk), t


def test_write_on_insertion_default_unchanged(tmp_path):
    """Default policy (on_insertion) behaves exactly as before Gap 5."""
    reader, source = _make_source(300, 100)
    ds = StreamCache(
        reader,
        memory_capacity=1,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
        write_policy="on_insertion",
    )
    ds.ingest_all()
    result = pa.RecordBatchReader.from_stream(ds).read_all()
    assert result.sort_by("id").equals(source.sort_by("id"))


def test_write_on_eviction_full_roundtrip(tmp_path):
    """WriteOnEviction with a tiny hot layer: everything is evicted to disk."""
    reader, source = _make_source(300, 100)
    ds = StreamCache(
        reader,
        memory_capacity=1,  # tiny hot → everything evicted to disk
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
        write_policy="on_eviction",
    )
    ds.ingest_all()
    result = pa.RecordBatchReader.from_stream(ds).read_all()
    assert result.sort_by("id").equals(source.sort_by("id"))


def test_write_on_eviction_large_hot_no_disk_io(tmp_path):
    """WriteOnEviction with ample hot: nothing is ever evicted, disk stays empty."""
    reader, source = _make_source(100, 100)
    ds = StreamCache(
        reader,
        memory_capacity=64 * 1024 * 1024,  # large — fits everything
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
        write_policy="on_eviction",
    )
    ds.ingest_all()

    cache_file = next(tmp_path.rglob("cache.arrow"), None)
    assert cache_file is not None
    assert cache_file.stat().st_size == 0  # nothing written to disk

    result = pa.RecordBatchReader.from_stream(ds).read_all()
    assert result.sort_by("id").equals(source.sort_by("id"))


def test_write_on_eviction_replayable(tmp_path):
    """WriteOnEviction caches can be replayed multiple times with the same result."""
    reader, source = _make_source(250, 50)
    ds = StreamCache(
        reader,
        memory_capacity=4 * 1024,  # small hot → partial eviction
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
        write_policy="on_eviction",
    )
    ds.ingest_all()
    first = pa.RecordBatchReader.from_stream(ds.reader(from_start=True)).read_all()
    second = pa.RecordBatchReader.from_stream(ds.reader(from_start=True)).read_all()
    assert first.equals(second)
    assert first.sort_by("id").equals(source.sort_by("id"))


def test_write_on_eviction_multiple_readers(tmp_path):
    """Two independent readers under WriteOnEviction both return correct data."""
    reader, source = _make_source(300, 100)
    ds = StreamCache(
        reader,
        memory_capacity=1,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
        write_policy="on_eviction",
    )
    ds.ingest_all()
    r1 = pa.RecordBatchReader.from_stream(ds.reader(from_start=True)).read_all()
    r2 = pa.RecordBatchReader.from_stream(ds.reader(from_start=True)).read_all()
    assert r1.equals(r2)
    assert r1.sort_by("id").equals(source.sort_by("id"))


def test_write_on_eviction_concurrent_reader(tmp_path):
    """A reader running on another thread always finds every batch (hot or disk),
    never a gap, while the eviction-driven disk writes happen."""
    reader, source = _make_source(500, 50)
    ds = StreamCache(
        reader,
        memory_capacity=2 * 1024,  # forces frequent eviction during reads
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
        write_policy="on_eviction",
    )
    ds.ingest_all()

    results = []

    def read_all():
        results.append(
            pa.RecordBatchReader.from_stream(ds.reader(from_start=True)).read_all()
        )

    threads = [threading.Thread(target=read_all) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(results) == 4
    for r in results:
        assert r.sort_by("id").equals(source.sort_by("id"))


def test_write_on_eviction_disk_capacity_enforced(tmp_path):
    """WriteOnEviction still enforces disk_capacity at eviction time."""
    reader, _ = _make_source(300, 100)
    ds = StreamCache(
        reader,
        memory_capacity=1,  # force eviction
        disk_path=str(tmp_path),
        disk_capacity=10,  # impossibly small
        write_policy="on_eviction",
    )
    with pytest.raises(MemoryError, match="capacity"):
        ds.ingest_all()


def test_invalid_write_policy_raises_value_error():
    """Unknown write_policy string raises ValueError mentioning the parameter."""
    source = pa.table({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="write_policy"):
        StreamCache(
            source.to_reader(),
            memory_capacity=1,
            disk_path="/tmp",
            disk_capacity=1024,
            write_policy="invalid_policy",
        )


def test_write_policy_ignored_for_memory_only_tier():
    """write_policy is accepted (and ignored) for a memory-only cache."""
    source = pa.table({"x": list(range(100))})
    ds = StreamCache(source.to_reader(max_chunksize=50), write_policy="on_eviction")
    assert ds.ingest_all() == 2
    result = pa.RecordBatchReader.from_stream(ds).read_all()
    assert result.equals(source)
