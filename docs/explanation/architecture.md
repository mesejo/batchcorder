# Architecture

## Overview

batchcorder is a Rust extension module for Python, built with
[pyo3](https://pyo3.rs) and [maturin](https://github.com/PyO3/maturin).

The core idea is simple: wrap any Arrow stream source in a two-tier hybrid
cache so that multiple independent readers can replay it concurrently without
re-reading from the upstream source.

```
Python API (CachedDataset / CachedDatasetReader)
  └─ Rust implementation (pyo3 extension module)
       └─ Foyer HybridCache  ←  memory LRU tier
                             ←  disk tier (direct I/O)
```

## Why this problem is hard

Arrow `RecordBatchReader` is a single-use stream backed by the
[Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html).
Once `get_next` returns `nullptr` the stream is exhausted and cannot be reset.
There is no rewind.

This is fine for a single-pass pipeline, but common workloads need multiple
passes over the same data:

- **DuckDB self-joins** — the planner opens the same virtual table twice, once
  per side of the join.
- **ASOF joins with CTEs** — a table may be scanned first inside a CTE and
  again in the outer query.
- **Training loops** — each epoch iterates the full dataset.

batchcorder solves this by consuming the upstream stream exactly once and
caching every batch, so any subsequent reader can replay from any position.

## Dependency choices

| Dependency | Role |
|---|---|
| **pyo3** | Rust ↔ Python FFI; exposes `CachedDataset` and `CachedDatasetReader` as Python classes |
| **maturin** | Build backend; compiles the Rust crate into a `_batchcorder.so` extension module |
| **pyo3-arrow** | Bridges the Arrow C Data Interface with pyo3 types |
| **Foyer** | Two-tier hybrid cache (memory LRU + disk via direct I/O) |
| **Tokio** | Async runtime required by Foyer's async API |
| **arrow-ipc** | Serialises `RecordBatch` to bytes for storage in the cache |

## Shared mutable state

All shared state lives in `Arc<Mutex<DatasetInner>>`.  The `Arc` lets multiple
`CachedDatasetReader` handles (and the owning `CachedDataset`) share the same
inner state without copying.  The `Mutex` serialises writes — specifically,
ingestion of new batches from the upstream source.

```
CachedDataset
  └─ Arc<Mutex<DatasetInner>>
       ├─ upstream: Option<Box<dyn RecordBatchReader>>   (None once exhausted)
       ├─ cache: HybridCache<u64, Bytes>                 (batch_index → IPC bytes)
       ├─ ingested_count: u64
       └─ consumer_positions: Vec<u64>                   (one per live reader)

CachedDatasetReader  (×N, independent)
  ├─ Arc<Mutex<DatasetInner>>   (shared with the dataset)
  └─ position: u64              (local read cursor)
```

Reads (fetching IPC bytes from the cache) happen **outside** the mutex so that
multiple readers can fetch different batches from disk concurrently.

## Ingestion lifecycle

Ingestion is lazy: a batch is pulled from the upstream source only when a
reader advances to a position that has not yet been ingested.

```
reader.next()
  1. lock(inner)
  2. if position >= ingested_count: call ingest_up_to(position)
       └─ upstream.next() [with GIL acquired]
       └─ cache.insert(idx, ipc_bytes)
  3. clone cache handle for idx
  4. unlock(inner)
  5. cache.get(idx).await   ← disk I/O here, no lock held
  6. deserialize IPC bytes → RecordBatch
```

Step 5 is deliberately outside the lock so concurrent readers can fetch from
disk in parallel even though ingestion is serialised.

## Concurrency model and GIL safety

DuckDB drives its scanner threads from C++; those threads have no Python thread
state and hold no Python locks when `get_next` is invoked.  The upstream reader
(`PyRecordBatchReader`) calls Python C-API functions and **requires the GIL**.

This creates a potential ABBA deadlock between the internal `Mutex` and the
GIL.  batchcorder avoids it with a single, global lock-ordering rule:

> `inner.lock()` is always acquired **before** the GIL.
> The GIL is **never** held while waiting for `inner.lock()`.

All `#[pymethods]` that access `inner` drop the GIL first
(`without_gil(py, || inner.lock())`).  `ingest_up_to`, which is called with
`inner.lock()` already held, re-acquires the GIL via `with_gil_acquired` only
when it needs to call the upstream Python reader.

See [GIL & locking design](gil-deadlock.md) for the full ABBA analysis and the
specific commit that introduced the fix.

## Cache eviction

Foyer evicts entries from the memory tier to the disk tier in the background
when `memory_capacity` is exceeded.  Entries are dropped entirely when
`disk_capacity` is exceeded.

If a reader requests a batch that has been fully evicted, `cache.get` returns
`None` and the reader raises:

```
Batch N was evicted from the cache before it could be read.
Increase cache capacity so it can hold all live consumer positions.
```

The minimum safe disk capacity is the serialised size of all batches that any
live reader might still need to read — i.e., the span between the slowest and
fastest concurrent reader.

## Arrow PyCapsule Interface

Both `CachedDataset` and `CachedDatasetReader` implement `__arrow_c_stream__`
and `__arrow_c_schema__`.  This means any Arrow-compatible library —
PyArrow, DuckDB, DataFusion, Polars — can consume them with zero-copy transfer
via the Arrow C Data Interface, without batchcorder needing to know anything
about the consumer.

Each call to `__arrow_c_stream__` on a `CachedDataset` creates a fresh
`CachedDatasetReader` starting at batch 0, enabling DuckDB to open the same
table multiple times in a single query.
