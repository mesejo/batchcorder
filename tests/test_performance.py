"""Performance and throughput guarantees for StreamCache.

These tests verify that:
  1. Upstream is consumed exactly once regardless of how many readers replay it.
  2. Cache hits do not touch the upstream source.
  3. Repeated reads are substantially faster than re-reading a slow upstream.
  4. Concurrent reads complete without full serialization.

Tests use a CountingReader / SlowReader rather than wall-clock assertions wherever
possible to avoid flakiness.  Timing tests use a deliberately slow upstream
(50 ms per batch) with a conservative 5x margin so they stay meaningful even
under CI load.
"""

import threading
import time

import pyarrow as pa

from batchcorder import StreamCache


# ── helpers ───────────────────────────────────────────────────────────────────


class CountingReader:
    """Arrow stream source that counts how many batches the upstream has yielded."""

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = batches
        self.batches_read = 0
        self._lock = threading.Lock()

    def __arrow_c_schema__(self):
        return self._batches[0].schema.__arrow_c_schema__()

    def __arrow_c_stream__(self, requested_schema=None):
        def _gen():
            for batch in self._batches:
                with self._lock:
                    self.batches_read += 1
                yield batch

        return pa.RecordBatchReader.from_batches(
            self._batches[0].schema, _gen()
        ).__arrow_c_stream__(requested_schema)


class SlowReader:
    """Arrow stream source that sleeps `delay` seconds before yielding each batch."""

    def __init__(self, batches: list[pa.RecordBatch], delay: float = 0.05) -> None:
        self._batches = batches
        self._delay = delay

    def __arrow_c_schema__(self):
        return self._batches[0].schema.__arrow_c_schema__()

    def __arrow_c_stream__(self, requested_schema=None):
        def _gen():
            for batch in self._batches:
                time.sleep(self._delay)
                yield batch

        return pa.RecordBatchReader.from_batches(
            self._batches[0].schema, _gen()
        ).__arrow_c_stream__(requested_schema)


def _make_batches(n_batches: int = 5, rows: int = 10) -> list[pa.RecordBatch]:
    schema = pa.schema([("id", pa.int64()), ("value", pa.float64())])
    return [
        pa.record_batch(
            {
                "id": list(range(i * rows, (i + 1) * rows)),
                "value": [float(j) for j in range(i * rows, (i + 1) * rows)],
            },
            schema=schema,
        )
        for i in range(n_batches)
    ]


# ── upstream read-once guarantee ──────────────────────────────────────────────


def test_upstream_read_once_with_sequential_readers():
    """K sequential readers collectively trigger the upstream exactly M times (not KxM)."""
    n_readers = 5
    batches = _make_batches(n_batches=6)
    source = CountingReader(batches)
    ds = StreamCache(source)

    for _ in range(n_readers):
        pa.RecordBatchReader.from_stream(ds.reader()).read_all()

    assert source.batches_read == len(batches), (
        f"Expected upstream called {len(batches)} times, got {source.batches_read}"
    )


def test_upstream_read_once_with_concurrent_readers():
    """K concurrent readers collectively trigger the upstream exactly M times (not KxM)."""
    n_readers = 8
    batches = _make_batches(n_batches=6)
    source = CountingReader(batches)
    ds = StreamCache(source)
    errors: list[Exception] = []

    def read():
        try:
            pa.RecordBatchReader.from_stream(ds.reader()).read_all()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=read) for _ in range(n_readers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert source.batches_read == len(batches), (
        f"Expected upstream called {len(batches)} times, got {source.batches_read}"
    )


def test_upstream_read_once_memory_only():
    """Same read-once guarantee for memory-only mode."""
    n_readers = 5
    batches = _make_batches(n_batches=4)
    source = CountingReader(batches)
    ds = StreamCache(source)

    for _ in range(n_readers):
        pa.RecordBatchReader.from_stream(ds.reader()).read_all()

    assert source.batches_read == len(batches)


def test_upstream_not_re_read_after_ingest_all():
    """After ingest_all(), reads do not increment the upstream counter."""
    batches = _make_batches(n_batches=5)
    source = CountingReader(batches)
    ds = StreamCache(source)

    ds.ingest_all()
    count_after_ingest = source.batches_read
    assert count_after_ingest == len(batches)

    # Three more reads — upstream must not be touched again.
    for _ in range(3):
        pa.RecordBatchReader.from_stream(ds.reader()).read_all()

    assert source.batches_read == count_after_ingest, (
        "Cache reads triggered additional upstream reads after ingest_all()"
    )


def test_ingested_count_stable_after_full_ingest():
    """ingested_count doesn't grow after upstream is exhausted, regardless of reads."""
    batches = _make_batches(n_batches=4)
    ds = StreamCache(
        pa.RecordBatchReader.from_batches(batches[0].schema, iter(batches))
    )
    total = ds.ingest_all()

    for _ in range(5):
        pa.RecordBatchReader.from_stream(ds.reader()).read_all()

    assert ds.ingested_count == total
    assert ds.upstream_exhausted


# ── cache-hit speed ───────────────────────────────────────────────────────────


def test_cache_hit_faster_than_upstream():
    """Reading from a warm cache is substantially faster than re-reading a slow upstream.

    Each batch has a 50 ms artificial delay.  The first read pays ingestion cost;
    the second read must complete in under 1/5 of that time (cache, no delay).
    """
    delay = 0.05  # 50 ms per batch
    n_batches = 4
    batches = _make_batches(n_batches=n_batches)
    source = SlowReader(batches, delay=delay)
    ds = StreamCache(source)

    # First read: pays ingestion cost (n_batches x delay).
    t0 = time.perf_counter()
    pa.RecordBatchReader.from_stream(ds.reader()).read_all()
    first_read_s = time.perf_counter() - t0

    # Second read: cache hits only, no upstream delay.
    t1 = time.perf_counter()
    pa.RecordBatchReader.from_stream(ds.reader()).read_all()
    second_read_s = time.perf_counter() - t1

    # Require at least 5x speedup — generous enough to survive a loaded CI box.
    assert second_read_s < first_read_s / 5, (
        f"Cache read ({second_read_s:.3f}s) not substantially faster than "
        f"ingestion read ({first_read_s:.3f}s)"
    )


def test_disk_cache_hit_faster_than_slow_upstream(tmp_path):
    """With disk spill, a second read is faster than re-ingesting a slow upstream."""
    delay = 0.05
    n_batches = 4
    batches = _make_batches(n_batches=n_batches)
    source = SlowReader(batches, delay=delay)
    ds = StreamCache(
        source,
        memory_capacity=1024,  # 1 KiB — forces everything to disk
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )

    t0 = time.perf_counter()
    pa.RecordBatchReader.from_stream(ds.reader()).read_all()
    first_read_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pa.RecordBatchReader.from_stream(ds.reader()).read_all()
    second_read_s = time.perf_counter() - t1

    assert second_read_s < first_read_s / 5, (
        f"Disk cache read ({second_read_s:.3f}s) not faster than "
        f"ingestion read ({first_read_s:.3f}s)"
    )


# ── large-scale stress ────────────────────────────────────────────────────────


def test_many_readers_many_batches_memory_only():
    """20 readers x 50 batches: all readers get complete, correct data."""
    n_readers = 20
    n_batches = 50
    batches = _make_batches(n_batches=n_batches, rows=5)
    source = CountingReader(batches)
    expected = pa.Table.from_batches(batches)
    ds = StreamCache(source)

    results: list[pa.Table | None] = [None] * n_readers
    errors: list[Exception] = []

    def read(i):
        try:
            results[i] = pa.RecordBatchReader.from_stream(ds.reader()).read_all()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=read, args=(i,)) for i in range(n_readers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    for res in results:
        assert res is not None
        assert res.equals(expected)

    # Upstream must have been read exactly once across all 20 readers.
    assert source.batches_read == n_batches
