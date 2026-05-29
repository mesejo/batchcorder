"""Error quality tests for StreamCache and StreamCacheReader.

Each test verifies three things for a given error condition:
  1. The correct exception *type* is raised (not a generic RuntimeError or similar).
  2. The error *message* is not mangled — keywords survive the Rust→Python boundary.
  3. The exception has a non-empty, human-readable *traceback* (i.e. __traceback__
     is set and format_exception produces something useful).
"""

import os
import traceback

import pyarrow as pa
import pytest

from batchcorder import StreamCache


# ── helpers ───────────────────────────────────────────────────────────────────


def _ds(tmp_path):
    table = pa.table({"x": [1, 2, 3]})
    return StreamCache(
        table.to_reader(max_chunksize=3),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )


def _consumed_reader(tmp_path):
    """Return a StreamCacheReader that has been consumed via __arrow_c_stream__."""
    r = _ds(tmp_path).reader()
    pa.RecordBatchReader.from_stream(r).read_all()
    assert r.closed
    return r


def _assert_readable_traceback(exc_info) -> str:
    """Check the exception has a populated, human-readable traceback.

    Returns the formatted traceback text so callers can make additional
    assertions without repeating the formatting logic.
    """
    exc = exc_info.value
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    text = "".join(lines)
    assert text.strip(), "traceback must not be empty"
    assert type(exc).__name__ in text, (
        f"exception type name '{type(exc).__name__}' missing from traceback:\n{text}"
    )
    return text


# Accessing a consumed reader raises ValueError (invalid state, not an I/O
# failure).  __next__ also raises ValueError rather than StopIteration when
# the reader has been consumed, not merely exhausted.


def test_reader_c_stream_consumed_raises_valueerror(tmp_path):
    r = _consumed_reader(tmp_path)
    with pytest.raises(ValueError, match="consumed") as exc_info:
        r.__arrow_c_stream__()
    _assert_readable_traceback(exc_info)


def test_reader_c_schema_consumed_raises_valueerror(tmp_path):
    r = _consumed_reader(tmp_path)
    with pytest.raises(ValueError, match="consumed") as exc_info:
        r.__arrow_c_schema__()
    _assert_readable_traceback(exc_info)


def test_reader_schema_property_consumed_raises_valueerror(tmp_path):
    r = _consumed_reader(tmp_path)
    with pytest.raises(ValueError, match="consumed") as exc_info:
        _ = r.schema
    _assert_readable_traceback(exc_info)


def test_reader_next_consumed_raises_valueerror(tmp_path):
    # next() on a consumed (not merely exhausted) reader raises ValueError, not
    # StopIteration.  Python's built-in next() propagates non-StopIteration
    # exceptions as-is, so the ValueError surfaces directly.
    r = _consumed_reader(tmp_path)
    with pytest.raises(ValueError, match="consumed") as exc_info:
        next(r)
    assert "consumed" in str(exc_info.value).lower()
    _assert_readable_traceback(exc_info)


# Exceptions raised inside the upstream Arrow C stream generator pass through
# pyo3-arrow's Rust error-conversion machinery.  Verify that the original
# message is not replaced by a generic "ArrowError" or similar.


class _ExplodingReader:
    """Yields one valid batch then raises ValueError to simulate a broken upstream."""

    _schema = pa.schema([("x", pa.int64())])

    def __arrow_c_stream__(self, requested_schema=None):
        def _gen():
            yield pa.record_batch({"x": pa.array([1, 2, 3])})
            raise ValueError("upstream exploded")

        return pa.RecordBatchReader.from_batches(
            self._schema, _gen()
        ).__arrow_c_stream__(requested_schema)


def test_upstream_error_message_not_mangled(tmp_path):
    """The upstream ValueError message must survive the Rust conversion boundary."""
    ds = StreamCache(
        _ExplodingReader(),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )
    with pytest.raises(ValueError, match="upstream exploded") as exc_info:
        pa.RecordBatchReader.from_stream(ds).read_all()

    # The original message must appear somewhere in the exception chain.
    # Walk __cause__ / __context__ in case the error is wrapped.
    full_text = str(exc_info.value)
    cause = exc_info.value.__cause__ or exc_info.value.__context__
    if cause is not None:
        full_text += " | " + str(cause)
    assert "upstream exploded" in full_text, (
        f"Original error message lost. Exception chain: {full_text}"
    )


def test_upstream_error_has_readable_traceback(tmp_path):
    """The traceback for an upstream error must be non-empty and name the exception."""
    ds = StreamCache(
        _ExplodingReader(),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )
    with pytest.raises(Exception, match="upstream exploded") as exc_info:
        pa.RecordBatchReader.from_stream(ds).read_all()

    text = _assert_readable_traceback(exc_info)
    assert "upstream exploded" in text, (
        f"Original error message not in traceback:\n{text}"
    )


# ── disk capacity enforcement (Gap 1) ─────────────────────────────────────────


def _make_disk_cache(tmp_path, subdir, disk_capacity, memory_capacity=1):
    """Helper: StreamCache forced to disk tier with controlled capacities."""
    table = pa.table(
        {
            "id": list(range(200)),
            "payload": pa.array([b"x" * 512] * 200, type=pa.large_binary()),
        }
    )
    return StreamCache(
        table.to_reader(max_chunksize=200),
        memory_capacity=memory_capacity,
        disk_path=str(tmp_path / subdir),
        disk_capacity=disk_capacity,
    )


def test_disk_capacity_exceeded_raises_memory_error(tmp_path):
    """disk_capacity smaller than one batch raises MemoryError on first ingest."""
    ds = _make_disk_cache(tmp_path, "a", disk_capacity=10)
    with pytest.raises(MemoryError, match="capacity") as exc_info:
        ds.ingest_all()
    _assert_readable_traceback(exc_info)


def test_disk_capacity_error_message_contains_sizes(tmp_path):
    """MemoryError message reports the capacity and bytes-written figures."""
    ds = _make_disk_cache(tmp_path, "b", disk_capacity=10)
    with pytest.raises(MemoryError) as exc_info:
        ds.ingest_all()
    msg = str(exc_info.value)
    # Capacity value (10) and "0 bytes already written" must appear.
    assert "10" in msg
    assert "0" in msg


def test_disk_capacity_ample_full_roundtrip(tmp_path):
    """Ample disk_capacity: full ingest + replay returns identical data."""
    table = pa.table({"x": list(range(300))})
    ds = StreamCache(
        table.to_reader(max_chunksize=100),
        memory_capacity=1,
        disk_path=str(tmp_path / "c"),
        disk_capacity=64 * 1024 * 1024,
    )
    assert ds.ingest_all() == 3
    result = pa.RecordBatchReader.from_stream(ds).read_all()
    assert result.equals(table)


def test_disk_capacity_exceeded_mid_stream(tmp_path):
    """disk_capacity that fits some but not all batches errors mid-ingest;
    already-ingested batches are still readable."""
    # Ingest one batch into a large-capacity cache to measure its on-disk size.
    probe = StreamCache(
        pa.table(
            {
                "id": list(range(100)),
                "payload": pa.array([b"y" * 512] * 100, type=pa.large_binary()),
            }
        ).to_reader(max_chunksize=100),
        memory_capacity=1,
        disk_path=str(tmp_path / "probe"),
        disk_capacity=64 * 1024 * 1024,
    )
    probe.ingest_all()
    cache_file = next((tmp_path / "probe").rglob("cache.arrow"))
    one_batch_bytes = cache_file.stat().st_size

    # Now use a capacity that fits exactly one batch.
    table = pa.table(
        {
            "id": list(range(300)),
            "payload": pa.array([b"y" * 512] * 300, type=pa.large_binary()),
        }
    )
    ds = StreamCache(
        table.to_reader(max_chunksize=100),
        memory_capacity=1,
        disk_path=str(tmp_path / "real"),
        disk_capacity=one_batch_bytes + 1,
    )
    with pytest.raises(MemoryError, match="capacity"):
        ds.ingest_all()
    assert ds.ingested_count >= 1


# ── Gap 3: checksum / corruption detection ────────────────────────────────────


def _disk_cache_checksums(tmp_path, subdir="cache") -> tuple:
    source = pa.table({"id": list(range(300)), "v": list(range(300))})
    ds = StreamCache(
        source.to_reader(max_chunksize=100),
        memory_capacity=1,  # force all to disk
        disk_path=str(tmp_path / subdir),
        disk_capacity=64 * 1024 * 1024,
    )
    return ds, source


def _find_cache_file(tmp_path, subdir="cache"):
    return next((tmp_path / subdir).rglob("cache.arrow"))


def test_checksum_valid_roundtrip(tmp_path):
    """Normal round-trip is unaffected by the checksum layer."""
    ds, source = _disk_cache_checksums(tmp_path)
    ds.ingest_all()
    result = pa.RecordBatchReader.from_stream(ds).read_all()
    assert result.sort_by("id").equals(source.sort_by("id"))


def test_checksum_mismatch_raises_ioerror(tmp_path):
    """Flipping a byte in the IPC payload raises OSError mentioning checksum."""
    ds, _ = _disk_cache_checksums(tmp_path)
    ds.ingest_all()

    cache_file = _find_cache_file(tmp_path)
    with open(cache_file, "r+b") as f:
        # Skip 8-byte checksum header for batch 0; corrupt byte 100 of payload.
        f.seek(8 + 100)
        original = f.read(1)
        f.seek(8 + 100)
        f.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(OSError, match=r"[Cc]hecksum|mismatch|corrupt"):
        pa.RecordBatchReader.from_stream(ds.reader(from_start=True)).read_all()


def test_checksum_truncated_file_raises_ioerror(tmp_path):
    """A truncated cache file raises OSError, not a silent short read."""
    ds, _ = _disk_cache_checksums(tmp_path)
    ds.ingest_all()

    cache_file = _find_cache_file(tmp_path)
    size = cache_file.stat().st_size
    os.truncate(cache_file, size // 2)

    with pytest.raises(OSError, match=r"[Dd]isk read|fill whole buffer|EOF|truncated"):
        pa.RecordBatchReader.from_stream(ds.reader(from_start=True)).read_all()


def test_checksum_header_corruption_raises_ioerror(tmp_path):
    """Corrupting the 8-byte checksum header raises OSError."""
    ds, _ = _disk_cache_checksums(tmp_path)
    ds.ingest_all()

    cache_file = _find_cache_file(tmp_path)
    with open(cache_file, "r+b") as f:
        f.seek(0)
        f.write(b"\x00" * 8)

    with pytest.raises(OSError, match=r"[Cc]hecksum|mismatch|corrupt"):
        pa.RecordBatchReader.from_stream(ds.reader(from_start=True)).read_all()


def test_checksum_hot_hit_bypasses_disk(tmp_path):
    """Batch served from hot layer skips disk; corrupting file doesn't affect hot reads."""
    source = pa.table({"id": list(range(100))})
    ds = StreamCache(
        source.to_reader(max_chunksize=100),
        memory_capacity=64 * 1024 * 1024,  # large hot — batch stays hot
        disk_path=str(tmp_path / "cache"),
        disk_capacity=64 * 1024 * 1024,
    )
    ds.ingest_all()

    cache_file = _find_cache_file(tmp_path, "cache")
    with open(cache_file, "r+b") as f:
        f.seek(8 + 100)
        f.write(b"\xff" * 10)

    result = pa.RecordBatchReader.from_stream(ds.reader(from_start=True)).read_all()
    assert result.equals(source)


# ── Gap 4: I/O error type mapping ─────────────────────────────────────────────


def test_capacity_error_is_memoryerror_not_ioerror(tmp_path):
    """Disk capacity exhaustion surfaces as MemoryError, not OSError."""
    source = pa.table({"x": list(range(100))})
    with pytest.raises(MemoryError, match="capacity"):
        StreamCache(
            source.to_reader(max_chunksize=100),
            memory_capacity=1,
            disk_path=str(tmp_path / "small"),
            disk_capacity=1,  # below one batch → capacity check fires
        ).ingest_all()


def test_disk_read_failure_raises_oserror(tmp_path):
    """A read failure (payload truncated away) raises OSError, not ArrowException."""
    ds, _ = _disk_cache_checksums(tmp_path, "read_fail")
    ds.ingest_all()

    cache_file = _find_cache_file(tmp_path, "read_fail")
    os.truncate(cache_file, 9)  # leave checksum header only, no payload

    try:
        pa.RecordBatchReader.from_stream(ds.reader(from_start=True)).read_all()
    except OSError:
        pass  # correct: short read maps to OSError
    except Exception as e:
        pytest.fail(f"Expected OSError, got {type(e).__name__}: {e}")
    else:
        pytest.fail("Expected OSError")
