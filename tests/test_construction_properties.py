"""Property-based tests for ``StreamCache`` construction and eviction policies.

The central contract: the storage parameters chosen at construction time —
hot-layer ``memory_capacity``, the memory/disk tier, and the ``write_policy``
(``on_insertion`` vs ``on_eviction``) — are *observationally transparent*.
They change where and when batches are stored and evicted, never the data a
reader replays.  These tests fix the source stream and vary the construction
parameters, asserting the replayed data is invariant.

Error paths (mismatched disk args, unknown policy) are covered by the
example-based tests in ``test_write_policy.py``; they are not @given here
because they are not part of the valid input space.
"""

from __future__ import annotations

import pyarrow as pa
from hypothesis import HealthCheck, given, settings

from batchcorder import StreamCache, StreamCacheReader
from tests.strategies import (
    construction_kwargs,
    disk_hot_capacities,
    memory_tier_capacities,
    table_and_chunksize,
    write_policies,
)


# Disk-backed examples serialise every batch through Arrow IPC, so the
# per-example cost is well above the 200ms default deadline for large streams.
# Disable the deadline and allow function-scoped tmp_path under @given.
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


def _replay(ds: StreamCache | StreamCacheReader) -> pa.Table:
    """Materialise a fresh full replay of the cache from batch 0."""
    return pa.RecordBatchReader.from_stream(ds).read_all()


class TestReplayInvariance:
    """The replayed stream equals the source for every valid construction."""

    @_settings
    @given(data=table_and_chunksize(), kwargs=construction_kwargs())
    def test_replay_preserves_source(self, data, kwargs, tmp_path):
        # Across the whole construction space (tier, hot capacity from
        # full-eviction to no-eviction, both write policies) a single replay
        # reproduces the source exactly, order included.
        table, chunksize = data
        ds = StreamCache(
            table.to_reader(max_chunksize=chunksize),
            **_resolve(kwargs, tmp_path),
        )
        assert _replay(ds).equals(table)

    @_settings
    @given(data=table_and_chunksize(), kwargs=construction_kwargs())
    def test_replay_idempotent(self, data, kwargs, tmp_path):
        # Two independent readers off the same cache yield identical data,
        # regardless of how much has been evicted to disk between them.
        table, chunksize = data
        ds = StreamCache(
            table.to_reader(max_chunksize=chunksize),
            **_resolve(kwargs, tmp_path),
        )
        first = _replay(ds.reader(from_start=True))
        second = _replay(ds.reader(from_start=True))
        assert first.equals(second)


class TestPolicyEquivalence:
    """Differential properties isolating a single construction dimension."""

    @_settings
    @given(data=table_and_chunksize(), memory_capacity=disk_hot_capacities)
    def test_write_policy_does_not_affect_data(self, data, memory_capacity, tmp_path):
        # Same source and same hot budget: on_insertion and on_eviction differ
        # only in when batches reach disk, never in the data replayed.  This is
        # the core guarantee of the eviction-time write path (Gap 5).
        table, chunksize = data
        disk_capacity = 256 * 1024 * 1024
        on_insertion = StreamCache(
            table.to_reader(max_chunksize=chunksize),
            write_policy="on_insertion",
            memory_capacity=memory_capacity,
            disk_path=str(tmp_path / "ins"),
            disk_capacity=disk_capacity,
        )
        on_eviction = StreamCache(
            table.to_reader(max_chunksize=chunksize),
            write_policy="on_eviction",
            memory_capacity=memory_capacity,
            disk_path=str(tmp_path / "evi"),
            disk_capacity=disk_capacity,
        )
        assert _replay(on_insertion).equals(_replay(on_eviction))

    @_settings
    @given(
        data=table_and_chunksize(),
        mem_capacity=memory_tier_capacities,
        hot_capacity=disk_hot_capacities,
        write_policy=write_policies,
    )
    def test_tier_does_not_affect_data(
        self, data, mem_capacity, hot_capacity, write_policy, tmp_path
    ):
        # A memory-only cache and a disk-backed cache built from the same
        # source replay identical data; the disk tier (and its eviction) is a
        # storage detail, not a data transform.  Each tier takes a capacity
        # from its own valid range (a hard cap for memory, a soft hot budget
        # for disk) — the invariant holds regardless of either.
        table, chunksize = data
        memory = StreamCache(
            table.to_reader(max_chunksize=chunksize),
            memory_capacity=mem_capacity,
        )
        disk = StreamCache(
            table.to_reader(max_chunksize=chunksize),
            memory_capacity=hot_capacity,
            disk_path=str(tmp_path),
            disk_capacity=256 * 1024 * 1024,
            write_policy=write_policy,
        )
        assert _replay(memory).equals(_replay(disk))
