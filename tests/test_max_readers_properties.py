"""Property-based tests for the ``max_readers`` bounded-replay contract.

The contract: ``max_readers=N`` makes low-water-mark eviction *observationally
transparent* to the N readers — every reader replays the source stream
exactly, regardless of storage tier, write policy, or how reader progress
interleaves — while enforcing N as a hard cap on reader creation and never
evicting before all N readers exist.

Example-based coverage of the same feature lives in ``test_max_readers.py``;
these tests explore the input space (table shapes, chunkings, reader counts,
interleavings, drop patterns) that fixed examples cannot.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings

from batchcorder import StreamCache
from tests.strategies import (
    cap_exhaustion_plan,
    construction_kwargs,
    interleaved_read_plan,
    max_readers_counts,
    partial_drop_plan,
    table_and_chunksize,
    undersubscribed_plan,
)


# Disk-backed examples serialise batches through Arrow IPC, and interleaved
# plans take many small steps per example; disable the deadline and allow the
# function-scoped tmp_path fixture under @given.
_settings = settings(
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _resolve(kwargs: dict, tmp_path) -> dict:
    """Replace the disk_path placeholder with a real temp directory."""
    resolved = dict(kwargs)
    if resolved.get("disk_path") == "<tmp_path>":
        resolved["disk_path"] = str(tmp_path)
    return resolved


def _create_reader(ds: StreamCache, op: str):
    """Perform one reader-creating operation on the cache."""
    if op == "reader":
        return ds.reader(from_start=True)
    if op == "reader_tail":
        return ds.reader(from_start=False)
    if op == "iter":
        return iter(ds)
    if op == "c_stream":
        return ds.__arrow_c_stream__()
    if op == "cast_stream":
        return pa.RecordBatchReader.from_stream(ds.cast(ds.schema))
    raise AssertionError(f"unknown op {op!r}")


# ── bounded replay invariance ────────────────────────────────────────────────


@_settings
@given(
    data=table_and_chunksize(),
    kwargs=construction_kwargs(),
    n_readers=max_readers_counts,
)
def test_each_reader_replays_source(data, kwargs, n_readers, tmp_path):
    # Sequential full reads: the first reader runs ahead while the rest sit at
    # batch 0, then each catch-up read trails the advancing low-water mark.
    # Every reader must still see the full source.
    table, chunksize = data
    ds = StreamCache(
        table.to_reader(max_chunksize=chunksize),
        max_readers=n_readers,
        **_resolve(kwargs, tmp_path),
    )
    readers = [ds.reader() for _ in range(n_readers)]
    for r in readers:
        assert pa.RecordBatchReader.from_stream(r).read_all().equals(table)


# ── interleaving invariance ──────────────────────────────────────────────────


@_settings
@given(plan=interleaved_read_plan(), kwargs=construction_kwargs())
def test_interleaved_readers_replay_source(plan, kwargs, tmp_path):
    # Drive all N readers through an arbitrary interleaving of batch pulls, so
    # eviction fires at every reachable low-water mark.  The batches collected
    # per reader must reassemble into the source.
    table, chunksize, n_readers, schedule = plan
    ds = StreamCache(
        table.to_reader(max_chunksize=chunksize),
        max_readers=n_readers,
        **_resolve(kwargs, tmp_path),
    )
    readers = [ds.reader() for _ in range(n_readers)]
    collected: list[list[pa.RecordBatch]] = [[] for _ in range(n_readers)]
    for i in schedule:
        collected[i].append(next(readers[i]))
    for batches in collected:
        assert pa.Table.from_batches(batches).equals(table)


# ── hard cap on reader creation ──────────────────────────────────────────────


@_settings
@given(plan=cap_exhaustion_plan())
def test_exactly_max_creations_allowed(plan):
    # Any mix of N creations succeeds; the (N+1)th raises ValueError no matter
    # which creation path is used and even though no reader was ever consumed
    # or dropped (dropping does not free a slot).
    table, n_readers, ops, extra_op = plan
    ds = StreamCache(table.to_reader(), max_readers=n_readers)
    held = [_create_reader(ds, op) for op in ops]  # all N must succeed
    with pytest.raises(ValueError, match="Maximum number of readers"):
        _create_reader(ds, extra_op)
    del held


# ── eviction gating ──────────────────────────────────────────────────────────


@_settings
@given(plan=undersubscribed_plan())
def test_undersubscribed_cache_retains_batch_zero(plan):
    # k < N readers fully consume the stream.  If eviction (incorrectly) ran,
    # batch 0 would be gone and a from-start reader would fail; the gating
    # contract says it must still replay the full source.
    table, chunksize, n_readers, k = plan
    ds = StreamCache(
        table.to_reader(max_chunksize=chunksize),
        max_readers=n_readers,
    )
    for _ in range(k):
        list(ds.reader())  # would advance the LWM if eviction were active
    replay = pa.RecordBatchReader.from_stream(ds.reader(from_start=True))
    assert replay.read_all().equals(table)


# ── dropped readers ──────────────────────────────────────────────────────────


@_settings
@given(plan=partial_drop_plan(), kwargs=construction_kwargs())
def test_survivors_replay_source_after_drops(plan, kwargs, tmp_path):
    # A strict subset of readers consume a prefix and drop their handles,
    # un-pinning the low-water mark.  The surviving readers — created before
    # any eviction — must still replay the full source.
    table, chunksize, n_readers, drop_prefixes = plan
    ds = StreamCache(
        table.to_reader(max_chunksize=chunksize),
        max_readers=n_readers,
        **_resolve(kwargs, tmp_path),
    )
    readers: list = [ds.reader() for _ in range(n_readers)]
    for i, prefix in drop_prefixes.items():
        for _ in range(prefix):
            next(readers[i])
        readers[i] = None  # drop the handle (refcount death, no GC needed)
    for r in readers:
        if r is not None:
            assert pa.RecordBatchReader.from_stream(r).read_all().equals(table)
