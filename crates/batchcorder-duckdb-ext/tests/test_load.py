"""End-to-end: LOAD the built extension into the Python DuckDB and tee a
pyarrow stream.

Needs the artifact from ``make duckdb-ext``; ``make test-duckdb-ext`` builds
it and runs these tests against every supported DuckDB version.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest


ROOT = Path(__file__).resolve().parents[3]
EXTENSION = ROOT / "target" / "duckdb-extension" / "batchcorder.duckdb_extension"

SCHEMA = pa.schema([("id", pa.int64()), ("m", pa.int64())])


class CountingStream:
    """A read-once pyarrow stream of `batches` batches that counts pulls."""

    def __init__(self, batches: int, rows: int) -> None:
        self.pulls = 0
        self._reader = pa.RecordBatchReader.from_batches(
            SCHEMA, self._batches(batches, rows)
        )
        # The capsule owns the ArrowArrayStream struct; tee_register moves
        # the stream out of it and leaves release = NULL, so the capsule's
        # destructor then has nothing to release.
        self._capsule = self._reader.__arrow_c_stream__()

    def _batches(self, batches: int, rows: int) -> Iterator[pa.RecordBatch]:
        for i in range(batches):
            self.pulls += 1
            values = pa.array(range(i * rows, (i + 1) * rows), pa.int64())
            yield pa.record_batch([values, values], schema=SCHEMA)

    @property
    def address(self) -> int:
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.restype = ctypes.c_void_p
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        address = get_pointer(self._capsule, b"arrow_array_stream")
        assert address is not None
        return address


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    if not EXTENSION.exists():
        pytest.fail(f"{EXTENSION} is missing; build it with `make duckdb-ext`")
    connection = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    connection.execute(f"LOAD '{EXTENSION}'")
    yield connection
    connection.close()


def test_functions_are_registered(con: duckdb.DuckDBPyConnection) -> None:
    names = {
        row[0]
        for row in con.execute(
            "SELECT function_name FROM duckdb_functions() WHERE function_name LIKE 'tee_%'"
        ).fetchall()
    }
    assert names == {"tee_register", "tee_scan", "tee_drop", "tee_stats"}


def test_self_join_reads_upstream_once(con: duckdb.DuckDBPyConnection) -> None:
    stream = CountingStream(batches=10, rows=1000)
    registered = con.execute(
        f"CALL tee_register('t', {stream.address}::UBIGINT)"
    ).fetchone()
    assert registered == ("t", 2)

    query = "SELECT count(*), sum(a.id) FROM {t} a JOIN {t} b ON a.m = b.id"
    con.execute(
        "CREATE TABLE reference AS SELECT range AS id, range AS m FROM range(10000)"
    )
    got = con.execute(query.format(t="tee_scan('t')")).fetchone()
    assert got == con.execute(query.format(t="reference")).fetchone()
    assert stream.pulls == 10


def test_sealed_tee_evicts_and_refuses_a_second_query(
    con: duckdb.DuckDBPyConnection,
) -> None:
    stream = CountingStream(batches=20, rows=1000)
    con.execute(f"CALL tee_register('t', {stream.address}::UBIGINT)")
    union = "SELECT count(*) FROM (SELECT * FROM tee_scan('t') UNION ALL SELECT * FROM tee_scan('t'))"
    assert con.execute(union).fetchone() == (40_000,)

    stats = con.execute("CALL tee_stats('t')").fetchone()
    assert stats is not None
    ingested, evicted, sealed = stats
    assert (ingested, sealed) == (20, True)
    assert evicted >= 18
    with pytest.raises(duckdb.Error, match="bounding := 'unbounded'"):
        con.execute(union).fetchone()
