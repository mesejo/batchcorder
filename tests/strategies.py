"""Hypothesis strategies for batchcorder.

Each strategy models the valid input space for ``StreamCache`` construction —
the source stream plus the storage/eviction parameters accepted by
``StreamCache.__init__`` (see ``python/batchcorder/__init__.py`` and the
constructor in ``src/cached_dataset.rs``).

The strategies stay strictly on the *happy path*: every value they produce is
one the constructor is designed to accept without raising.  Error paths
(mismatched disk_path/disk_capacity, unknown write_policy, negative capacity)
are exercised by explicit example-based tests, not by ``@given``.
"""

from __future__ import annotations

import pyarrow as pa
from hypothesis import strategies as st


# -- Atomic strategies ------------------------------------------------------

# The two accepted write policies (src/cached_dataset.rs:982).  Both apply only
# to the disk tier; the constructor still accepts either for a memory-only
# cache (where it is ignored).
write_policies = st.sampled_from(["on_insertion", "on_eviction"])

# A disk budget large enough that any table the table strategy can generate
# fits comfortably, so disk_capacity is never the binding constraint and we
# stay on the happy path.  256 MiB >> the few-MiB ceiling of `arrow_table`.
LARGE_DISK_CAPACITY = 256 * 1024 * 1024

# Disk hot-layer budget.  A *soft* eviction budget (src/cached_dataset.rs:395):
# the range spans from 1 byte (every batch evicted to disk immediately) up to
# 64 MiB (the whole stream fits in the hot layer and nothing is ever evicted),
# driving eviction across its full 0%..100% range.  ``None`` uses the
# system-RAM default.
disk_hot_capacities = st.one_of(
    st.none(),
    st.integers(min_value=1, max_value=64 * 1024 * 1024),
)

# Memory-only tier capacity.  Unlike the disk hot layer this is a *hard cap*
# (src/cached_dataset.rs:385): inserting a batch that would exceed it raises
# ``MemoryError`` rather than evicting.  So the valid happy-path range is only
# values large enough to hold any table ``arrow_table`` can produce (well under
# 1 MiB), plus ``None`` for the default.  64 MiB is a comfortable floor.
memory_tier_capacities = st.one_of(
    st.none(),
    st.integers(min_value=64 * 1024 * 1024, max_value=256 * 1024 * 1024),
)

# UTF-8 text without surrogate code points (Arrow utf8 rejects lone
# surrogates) and column names that are valid, distinct identifiers.
_utf8_text = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20)
_column_names = st.from_regex(r"col_[a-z]{1,8}", fullmatch=True)


# -- Column data strategies -------------------------------------------------

# Each entry maps a value strategy to its Arrow type.  Floats exclude NaN/inf
# so that ``Table.equals`` (which treats NaN as unequal to itself) reflects
# cache fidelity rather than float quirks.
_COLUMN_KINDS = [
    (st.integers(min_value=-(2**63), max_value=2**63 - 1), pa.int64()),
    (
        st.floats(allow_nan=False, allow_infinity=False, width=64),
        pa.float64(),
    ),
    (_utf8_text, pa.string()),
    (st.booleans(), pa.bool_()),
]


# -- Composed / domain strategies -------------------------------------------


@st.composite
def arrow_table(draw: st.DrawFn) -> pa.Table:
    """A non-empty Arrow table with 1-4 typed columns and 1-2000 rows.

    Models a realistic batchcorder source: several columns of mixed primitive
    types, sized small enough to fit the disk budget yet large enough that
    chunking produces multiple batches to evict.
    """
    n_rows = draw(st.integers(min_value=1, max_value=2000))
    n_cols = draw(st.integers(min_value=1, max_value=4))
    names = draw(st.lists(_column_names, min_size=n_cols, max_size=n_cols, unique=True))

    columns: dict[str, pa.Array] = {}
    for name in names:
        value_strategy, arrow_type = draw(st.sampled_from(_COLUMN_KINDS))
        values = draw(st.lists(value_strategy, min_size=n_rows, max_size=n_rows))
        columns[name] = pa.array(values, type=arrow_type)
    return pa.table(columns)


@st.composite
def table_and_chunksize(draw: st.DrawFn) -> tuple[pa.Table, int]:
    """A table paired with a ``max_chunksize`` for ``Table.to_reader``.

    The chunk size ranges from 1 (one row per batch — maximum batch count,
    maximum eviction churn) up to the full row count (a single batch), so the
    cache is exercised across stream granularities.  Returned as data rather
    than a live reader because each ``StreamCache`` consumes its reader exactly
    once; tests build a fresh reader per cache via ``table.to_reader``.
    """
    table = draw(arrow_table())
    chunksize = draw(st.integers(min_value=1, max_value=table.num_rows))
    return table, chunksize


# -- max_readers strategies ---------------------------------------------------

# Reader-count cap (``max_readers``).  Small counts keep interleaving schedules
# tractable while covering the single-reader and several-reader regimes; the
# constructor accepts any value >= 1 (``register_reader`` in
# src/cached_dataset.rs enforces the cap, not the magnitude).
max_readers_counts = st.integers(min_value=1, max_value=5)

# Every public operation that creates — and therefore counts against the cap —
# a reader: direct handles (both start positions), iteration, raw C-stream
# export, and reading a cast view (which registers a reader at stream-export
# time like the others).
reader_creating_ops = st.sampled_from(
    ["reader", "reader_tail", "iter", "c_stream", "cast_stream"]
)


def _chunked(draw: st.DrawFn, table: pa.Table) -> tuple[int, int]:
    """Draw a chunk size targeting <= 24 batches; return (chunksize, n_batches).

    Interleaving schedules grow linearly with the batch count, so the target
    keeps them short while still producing multi-batch streams that exercise
    low-water-mark eviction.
    """
    target_batches = draw(st.integers(min_value=1, max_value=24))
    chunksize = -(-table.num_rows // target_batches)
    n_batches = -(-table.num_rows // chunksize)
    return chunksize, n_batches


@st.composite
def interleaved_read_plan(draw: st.DrawFn) -> tuple[pa.Table, int, int, list[int]]:
    """A (table, chunksize, n_readers, schedule) tuple for eviction tests.

    ``schedule`` is a permutation of each reader index repeated once per
    batch: following it consumes every reader exactly to exhaustion while the
    readers' relative progress varies arbitrarily, so the low-water mark (and
    therefore eviction) advances through arbitrary reachable orderings.
    """
    table = draw(arrow_table())
    chunksize, n_batches = _chunked(draw, table)
    n_readers = draw(max_readers_counts)
    schedule = draw(
        st.permutations([r for r in range(n_readers) for _ in range(n_batches)])
    )
    return table, chunksize, n_readers, schedule


@st.composite
def partial_drop_plan(draw: st.DrawFn) -> tuple[pa.Table, int, int, dict[int, int]]:
    """A (table, chunksize, n_readers, drop_prefixes) tuple.

    ``drop_prefixes`` maps a strict subset of reader indices to the number of
    batches each consumes before its handle is dropped (0..n_batches).  At
    least one reader survives: once the dropped readers stop pinning the
    low-water mark, the survivors must still replay the full stream.
    """
    table = draw(arrow_table())
    chunksize, n_batches = _chunked(draw, table)
    n_readers = draw(st.integers(min_value=2, max_value=5))
    n_dropped = draw(st.integers(min_value=1, max_value=n_readers - 1))
    dropped = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_readers - 1),
            min_size=n_dropped,
            max_size=n_dropped,
            unique=True,
        )
    )
    prefixes = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_batches),
            min_size=n_dropped,
            max_size=n_dropped,
        )
    )
    return table, chunksize, n_readers, dict(zip(dropped, prefixes, strict=True))


@st.composite
def cap_exhaustion_plan(draw: st.DrawFn) -> tuple[pa.Table, int, list[str], str]:
    """A (table, n_readers, ops, extra_op) tuple for the hard-cap contract.

    ``ops`` is a mix of ``n_readers`` reader-creating operations that must all
    succeed; ``extra_op`` is the one-past-the-cap creation that must raise.
    No batches are consumed, so eviction never interferes with the cap check.
    """
    table = draw(arrow_table())
    n_readers = draw(max_readers_counts)
    ops = draw(st.lists(reader_creating_ops, min_size=n_readers, max_size=n_readers))
    extra_op = draw(reader_creating_ops)
    return table, n_readers, ops, extra_op


@st.composite
def undersubscribed_plan(draw: st.DrawFn) -> tuple[pa.Table, int, int, int]:
    """A (table, chunksize, max_readers, k) tuple with k < max_readers.

    Models a cache where only ``k`` of the promised ``max_readers`` readers
    have been created and fully consumed — the regime where eviction must NOT
    have started, because future readers may still replay from batch 0.
    """
    table = draw(arrow_table())
    chunksize, _ = _chunked(draw, table)
    n_readers = draw(st.integers(min_value=2, max_value=5))
    k = draw(st.integers(min_value=1, max_value=n_readers - 1))
    return table, chunksize, n_readers, k


@st.composite
def construction_kwargs(draw: st.DrawFn) -> dict:
    """Valid keyword arguments for ``StreamCache.__init__`` (minus ``reader``).

    Picks a memory-only or disk-backed tier with a 1:3 bias toward disk, since
    the eviction policies under test only operate on the disk tier.  The
    capacity is drawn from the range valid for the chosen tier (a soft budget
    for disk, a hard cap that must fit the data for memory-only).  The
    ``disk_path`` placeholder is filled in by the test from a ``tmp_path``
    fixture (Hypothesis cannot create temp directories).
    """
    disk = draw(st.sampled_from([False, True, True, True]))
    if disk:
        return {
            "memory_capacity": draw(disk_hot_capacities),
            "write_policy": draw(write_policies),
            "disk_path": "<tmp_path>",  # test substitutes a real directory
            "disk_capacity": LARGE_DISK_CAPACITY,
        }
    return {
        "memory_capacity": draw(memory_tier_capacities),
        # write_policy is accepted but ignored for a memory-only cache.
        "write_policy": draw(write_policies),
    }
