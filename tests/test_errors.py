"""Error quality tests for CachedDataset and CachedDatasetReader.

Each test verifies three things for a given error condition:
  1. The correct exception *type* is raised (not a generic RuntimeError or similar).
  2. The error *message* is not mangled — keywords survive the Rust→Python boundary.
  3. The exception has a non-empty, human-readable *traceback* (i.e. __traceback__
     is set and format_exception produces something useful).
"""

import traceback

import pyarrow as pa
import pytest

from batchcorder import CachedDataset


# ── helpers ───────────────────────────────────────────────────────────────────


def _ds(tmp_path):
    table = pa.table({"x": [1, 2, 3]})
    return CachedDataset(
        table.to_reader(max_chunksize=3),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )


def _consumed_reader(tmp_path):
    """Return a CachedDatasetReader that has been consumed via __arrow_c_stream__."""
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


# The stub documents IOError for __arrow_c_stream__, __arrow_c_schema__, and
# .schema.  __next__ also raises IOError (from the same Rust arm) rather than
# StopIteration when the reader has been consumed, not merely exhausted.


def test_reader_c_stream_consumed_raises_ioerror(tmp_path):
    r = _consumed_reader(tmp_path)
    with pytest.raises(OSError) as exc_info:
        r.__arrow_c_stream__()
    assert "consumed" in str(exc_info.value).lower()
    _assert_readable_traceback(exc_info)


def test_reader_c_schema_consumed_raises_ioerror(tmp_path):
    r = _consumed_reader(tmp_path)
    with pytest.raises(OSError) as exc_info:
        r.__arrow_c_schema__()
    assert "consumed" in str(exc_info.value).lower()
    _assert_readable_traceback(exc_info)


def test_reader_schema_property_consumed_raises_ioerror(tmp_path):
    r = _consumed_reader(tmp_path)
    with pytest.raises(OSError) as exc_info:
        _ = r.schema
    assert "consumed" in str(exc_info.value).lower()
    _assert_readable_traceback(exc_info)


def test_reader_next_consumed_raises_ioerror(tmp_path):
    # next() on a consumed (not merely exhausted) reader raises OSError, not
    # StopIteration.  Python's built-in next() propagates non-StopIteration
    # exceptions as-is, so the OSError surfaces directly.
    r = _consumed_reader(tmp_path)
    with pytest.raises(OSError) as exc_info:
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
    ds = CachedDataset(
        _ExplodingReader(),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )
    with pytest.raises(ValueError) as exc_info:
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
    ds = CachedDataset(
        _ExplodingReader(),
        memory_capacity=16 * 1024 * 1024,
        disk_path=str(tmp_path),
        disk_capacity=64 * 1024 * 1024,
    )
    with pytest.raises(Exception) as exc_info:
        pa.RecordBatchReader.from_stream(ds).read_all()

    text = _assert_readable_traceback(exc_info)
    assert "upstream exploded" in text, (
        f"Original error message not in traceback:\n{text}"
    )
