"""Tests for the batchcorder-datafusion wheel (TeeTableProvider over datafusion-ffi).

Needs the extension (``maturin develop --uv -m
crates/batchcorder-datafusion-py/Cargo.toml``) and the ``datafusion``
dependency group (``uv sync --group datafusion``).
"""

import gc
import re
import subprocess
import sys
import types
from importlib.metadata import PackageNotFoundError

import pyarrow as pa
import pytest
from datafusion import SessionContext

from batchcorder_datafusion import TeeTableProvider, _hosts


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
    ctx = _tee_ctx(provider)
    result = ctx.sql(sql).to_arrow_table()
    assert result.equals(_reference_result(sql))
    _assert_fanout(provider, scans)


def _assert_fanout(provider: TeeTableProvider, scans: int) -> None:
    assert provider.scans_planned == scans
    assert provider.scans_executed == scans
    assert provider.is_sealed
    assert provider.evicted_count == provider.ingested_count == N_ROWS // CHUNK


# A query must keep working once the host SessionContext is gone, as long as
# its DataFrame is alive -- as it does for in-memory tables.  The provider's
# FFI codec used to reach the host session through a weak reference that is
# resolved on every scan.
_SINGLE_SCAN = "select count(*) as c, sum(x) as s from t"
_SELF_JOIN = (
    "select a.k, count(*) as c from t a join t b on a.k = b.k group by a.k order by a.k"
)


def test_query_runs_after_context_is_deleted():
    provider = _provider()
    ctx = _tee_ctx(provider)
    df = ctx.sql(_SINGLE_SCAN)
    del ctx
    gc.collect()
    assert df.to_arrow_table().equals(_reference_result(_SINGLE_SCAN))
    _assert_fanout(provider, 1)


@pytest.mark.parametrize(
    ("sql", "scans"), [(_SINGLE_SCAN, 1), (_SELF_JOIN, 2)], ids=["single", "join"]
)
def test_query_on_temporary_context(sql, scans):
    provider = _provider()
    df = _tee_ctx(provider).sql(sql)
    gc.collect()
    assert df.to_arrow_table().equals(_reference_result(sql))
    _assert_fanout(provider, scans)


def test_stream_drains_after_context_is_deleted():
    provider = _provider()
    ctx = _tee_ctx(provider)
    stream = ctx.sql(_SELF_JOIN).execute_stream()
    del ctx
    gc.collect()
    rows = sum(batch.to_pyarrow().num_rows for batch in stream)
    assert rows == _reference_result(_SELF_JOIN).num_rows
    _assert_fanout(provider, 2)


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


# Host version guard (batchcorder_datafusion._hosts).


def _fake_versions(installed: dict[str, str]):
    def get_version(dist: str) -> str:
        if dist not in installed:
            raise PackageNotFoundError(dist)
        return installed[dist]

    return get_version


def _session_from(module: str) -> object:
    # An instance of a class defined in `module`, standing in for a host's
    # SessionContext.
    return type("SessionContext", (), {"__module__": module})()


# Versions are built from SUPPORTED_HOSTS, so moving a host to a new range
# (a new DataFusion major) needs no change here.


def _bounds(dist: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return _hosts.SUPPORTED_HOSTS[dist]


def _inside(dist: str, *tail: int, suffix: str = "") -> str:
    # A supported version: the lower bound, extended by `tail` and `suffix`
    # (datafusion: 55 -> "55.9.1"; xorq-datafusion: 0.3 -> "0.3.12rc1").
    low, _ = _bounds(dist)
    return _hosts._format(low + tail) + suffix


def _above(dist: str) -> str:
    # The (exclusive) upper bound: the first unsupported version.
    _, high = _bounds(dist)
    return _hosts._format(high)


def _below(dist: str) -> str:
    # The last version before the lower bound: 55 -> 54.99, 0.3 -> 0.2.99.
    low, _ = _bounds(dist)
    last = max(i for i, part in enumerate(low) if part > 0)
    return _hosts._format((*low[:last], low[last] - 1, 99))


def _needs(dist: str) -> str:
    low, high = _bounds(dist)
    return re.escape(f"{dist}>={_hosts._format(low)},<{_hosts._format(high)}")


@pytest.mark.parametrize(
    ("module", "installed"),
    [
        ("datafusion", {"datafusion": _inside("datafusion", 0, 0)}),
        ("datafusion.context", {"datafusion": _inside("datafusion", 9, 1)}),
        ("xorq_datafusion", {"xorq-datafusion": _inside("xorq-datafusion", 0)}),
        (
            "xorq_datafusion._internal",
            {"xorq-datafusion": _inside("xorq-datafusion", 12, suffix="rc1")},
        ),
        # Host identified but not installed as a distribution: nothing to check.
        ("datafusion", {}),
        # Unknown host: left to the datafusion-ffi version check.
        ("some_other_host", {"datafusion": _above("datafusion")}),
    ],
)
def test_check_host_accepts(module, installed):
    _hosts.check_host(_session_from(module), _fake_versions(installed))


@pytest.mark.parametrize(
    ("module", "dist", "installed"),
    [
        ("datafusion", "datafusion", _above("datafusion")),
        ("datafusion", "datafusion", _below("datafusion")),
        ("xorq_datafusion", "xorq-datafusion", _below("xorq-datafusion")),
        ("xorq_datafusion", "xorq-datafusion", _above("xorq-datafusion")),
    ],
)
def test_check_host_rejects(module, dist, installed):
    match = f"{re.escape(f'{dist} {installed}')}.*{_needs(dist)}"
    with pytest.raises(RuntimeError, match=match):
        _hosts.check_host(_session_from(module), _fake_versions({dist: installed}))


def _native_host_session(monkeypatch, package: str) -> object:
    # Like xorq-datafusion's SessionContext: a native class whose module name
    # (`let`) is not its package's, exported by the (fake) host package.
    cls = type("SessionContext", (), {"__module__": "let"})
    host = types.ModuleType(package)
    host.SessionContext = cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, package, host)
    return cls()


def test_check_host_identifies_host_by_exported_class(monkeypatch):
    dist = "xorq-datafusion"
    session = _native_host_session(monkeypatch, "xorq_datafusion")
    _hosts.check_host(session, _fake_versions({dist: _inside(dist)}))
    with pytest.raises(RuntimeError, match=_needs(dist)):
        _hosts.check_host(session, _fake_versions({dist: _above(dist)}))


def test_check_host_ignores_class_no_host_exports(monkeypatch):
    # The same native class, exported by a package that is not a known host:
    # even an unsupported xorq-datafusion is not checked against it.
    session = _native_host_session(monkeypatch, "some_other_host")
    _hosts.check_host(
        session, _fake_versions({"xorq-datafusion": _above("xorq-datafusion")})
    )


def test_check_host_ignores_codec_capsule():
    # A codec capsule does not say which host it came from.
    capsule = SessionContext().__datafusion_logical_extension_codec__()
    _hosts.check_host(capsule, _fake_versions({"datafusion": _above("datafusion")}))


def test_register_with_mismatched_host_raises(monkeypatch):
    # The installed datafusion (inside its range) against a table that
    # expects the next major: the real registration path must fail with the
    # guard's error.
    _, high = _bounds("datafusion")
    next_range = (high, (*high[:-1], high[-1] + 1))
    monkeypatch.setattr(
        _hosts,
        "SUPPORTED_HOSTS",
        {**_hosts.SUPPORTED_HOSTS, "datafusion": next_range},
    )
    ctx = SessionContext()
    with pytest.raises(RuntimeError, match=f"needs {_needs('datafusion')}"):
        ctx.register_table("t", _provider())
