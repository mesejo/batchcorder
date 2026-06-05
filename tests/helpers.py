"""Shared constants for the batchcorder test suite."""

import pyarrow as pa


# The full exception surface of the batchcorder boundary (`BoundaryError` in
# src/cached_dataset.rs): worker threads collect these so `assert not errors`
# reports cache failures.  Anything else escaping a worker is a test bug; it
# kills the thread (pytest only prints the traceback), so tests without
# result-based asserts must also verify that every worker completed.
CACHE_ERRORS = (ValueError, OSError, MemoryError, RuntimeError, pa.ArrowException)
