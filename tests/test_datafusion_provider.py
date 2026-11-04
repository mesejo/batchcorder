"""Tests for the batchcorder-datafusion wheel (TeeTableProvider over datafusion-ffi).

Skipped unless both datafusion-python and the batchcorder_datafusion extension
are installed (``maturin develop -m crates/batchcorder-datafusion-py/Cargo.toml``).
"""

import subprocess
import sys

import pyarrow as pa
import pytest


datafusion = pytest.importorskip("datafusion")
batchcorder_datafusion = pytest.importorskip("batchcorder_datafusion")

from batchcorder_datafusion import TeeTableProvider
from datafusion import SessionContext


N_ROWS = 1000
CHUNK = 50


def _batches() -> list[pa.RecordBatch]:
    # Independent batches, not slices of one table: the cache's memory
    # accounting charges a slice for its whole parent buffer.
    out = []
    for start in range(0, N_ROWS, CHUNK):
        x = list(range(start, start + CHUNK))
        out.append(pa.record_batch({"k": [v % 7 for v in x], "x": x}))
    return out


def _table() -> pa.Table:
    return pa.Table.from_batches(_batches())


def _provider(**kwargs) -> TeeTableProvider:
    reader = pa.RecordBatchReader.from_batches(_table().schema, _batches())
    return TeeTableProvider(reader, **kwargs)


def _tee_ctx(provider: TeeTableProvider) -> SessionContext:
    ctx = SessionContext()
    ctx.register_table("t", provider)
    return ctx


def _reference_result(sql: str) -> pa.Table:
    ctx = SessionContext()
    ctx.register_record_batches("t", [_batches()])
    result = ctx.sql(sql).to_arrow_table()
    return result


QUERIES = [
    pytest.param("select count(*) as c, sum(x) as s from t", 1, id="single-scan"),
    pytest.param("select x from t where k = 3 order by x", 1, id="filter"),
    pytest.param(
        "select a.k, count(*) as c from t a join t b on a.k = b.k group by a.k order by a.k",
        2,
        id="self-join",
    ),
    pytest.param(
        "select sum(x) as s, count(*) as c from (select x from t union all select x from t)",
        2,
        id="union-all",
    ),
    pytest.param(
        "with c as (select k, x from t where x % 2 = 0) "
        "select (select count(*) from c) as n, (select sum(x) from c) as s, "
        "(select max(k) from c) as m",
        3,
        id="cte-x3",
    ),
    pytest.param(
        "select count(*) as c from (select x from t union all select x from t where false)",
        1,
        id="pruned-branch",
    ),
]


@pytest.mark.parametrize(("sql", "scans"), QUERIES)
def test_query_matches_reference_with_exact_fanout(sql, scans):
    provider = _provider()
    # Keep the context alive until the query finishes: over FFI the provider
    # only holds a weak reference to the host session.
    ctx = _tee_ctx(provider)
    result = ctx.sql(sql).to_arrow_table()
    assert result.equals(_reference_result(sql))
    assert provider.scans_planned == scans
    assert provider.scans_executed == scans
    assert provider.is_sealed
    assert provider.evicted_count == provider.ingested_count == N_ROWS // CHUNK


def test_explain_then_execute():
    provider = _provider()
    ctx = _tee_ctx(provider)
    sql = "select count(*) as c from t a join t b on a.k = b.k"
    ctx.sql(f"explain {sql}").collect()
    assert provider.ingested_count == 0
    assert not provider.is_sealed
    assert ctx.sql(sql).to_arrow_table().equals(_reference_result(sql))
    assert provider.scans_planned == 4  # 2 for EXPLAIN, 2 for the query
    assert provider.scans_executed == 2


def test_sealed_provider_rejects_second_query_after_eviction():
    ctx = _tee_ctx(_provider())
    ctx.sql("select sum(x) from t").collect()
    with pytest.raises(Exception, match="register a new provider"):
        ctx.sql("select sum(x) from t").collect()


def test_unbounded_provider_can_be_queried_repeatedly():
    ctx = _tee_ctx(_provider(bounding="unbounded"))
    sql = "select k, sum(x) as s from t group by k order by k"
    expected = _reference_result(sql)
    for _ in range(3):
        assert ctx.sql(sql).to_arrow_table().equals(expected)


def test_bounded_memory_single_scan():
    # A few batches of budget (Rust counts allocated buffer sizes, which are a
    # little larger than ``nbytes``) against 20 batches in total: only works
    # if batches are evicted as the scan passes them.
    batch_bytes = _batches()[0].nbytes
    provider = _provider(memory_capacity=batch_bytes * 6)
    ctx = _tee_ctx(provider)
    sql = "select count(*) as c, sum(x) as s from t"
    assert ctx.sql(sql).to_arrow_table().equals(_reference_result(sql))


def test_disk_tier_self_join(tmp_path):
    provider = _provider(
        memory_capacity=1, disk_path=str(tmp_path), disk_capacity=1 << 26
    )
    ctx = _tee_ctx(provider)
    sql = "select a.k, count(*) as c from t a join t b on a.x = b.x group by a.k order by a.k"
    assert ctx.sql(sql).to_arrow_table().equals(_reference_result(sql))
    provider.close()


def test_schema_and_metadata():
    provider = _provider()
    assert provider.schema == _table().schema
    assert provider.scans_planned == 0
    assert not provider.upstream_exhausted


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bounding": "counted"}, "bounding must be"),
        ({"write_policy": "sometimes"}, "write_policy must be"),
        ({"disk_path": "/tmp"}, "disk_path and disk_capacity"),
    ],
)
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _provider(**kwargs)


# A Python-generator upstream read concurrently by several DataFusion scans
# on threads the host does not own.  Every pull needs the GIL, so this
# exercises the GIL-safe ingestion and the wait hook through the FFI
# boundary.  In a subprocess so a deadlock fails on timeout, not by hanging.
_PYTHON_UPSTREAM_SCRIPT = """
import pyarrow as pa
from datafusion import SessionContext
from batchcorder_datafusion import TeeTableProvider

schema = pa.schema([("k", pa.int64()), ("x", pa.int64())])

def gen():
    for b in range(200):
        x = list(range(b * 10, (b + 1) * 10))
        yield pa.record_batch({"k": [v % 5 for v in x], "x": x}, schema=schema)

provider = TeeTableProvider(pa.RecordBatchReader.from_batches(schema, gen()))
ctx = SessionContext()
ctx.register_table("t", provider)
out = ctx.sql(
    "select (select count(*) from t) as n, (select sum(x) from t) as s, "
    "(select count(*) from t a join t b on a.x = b.x) as j"
).to_pydict()
assert out == {"n": [2000], "s": [sum(range(2000))], "j": [2000]}, out
assert provider.scans_planned == 4, provider.scans_planned
print("ok")
"""


def test_python_upstream_concurrent_scans_do_not_deadlock():
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PYTHON_UPSTREAM_SCRIPT],
            timeout=60,
            check=False,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("deadlock: concurrent scans over a Python upstream hung")
    else:
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "ok"
