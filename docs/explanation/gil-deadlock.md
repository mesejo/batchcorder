# GIL / Mutex Deadlock Analysis and Fix

## Background

`CachedDataset` implements the Arrow C Data Interface so that DuckDB (and
other consumers) can call `__arrow_c_stream__` to open a replayable scan.
DuckDB drives those scans from its own C++ threads — threads that have no
Python thread state and hold no Python locks when `get_next` is invoked.

The implementation stores batches in a Foyer hybrid cache.  Ingestion from
the upstream Python reader is done lazily: the first call to `get_next` that
needs batch *N* acquires an internal `Mutex<DatasetInner>` and calls
`upstream.next()` to pull the batch from Python.

## What caused the deadlock

### Two locks in play

| Lock | Holder | Purpose |
|------|--------|---------|
| Python GIL | one thread at a time | required to call any Python C-API function |
| `inner` Mutex | one thread at a time | serialises ingestion and cache writes |

### The ABBA pattern

An ABBA deadlock occurs when two threads acquire the same two locks in
**opposite order**:

```
Thread A:  lock(X) → wait(Y)
Thread B:  lock(Y) → wait(X)
```

In the original code the two acquisition orders were:

**Scanner thread** (DuckDB calls `get_next` from a C++ thread — no GIL):

```
lock(inner)               ← acquired first
  └─ with_gil_acquired()  ← tries to acquire GIL second
       └─ upstream.next() ← needs GIL to call Python iterator
```

**Executor thread** (DuckDB calls `__arrow_c_stream__` to open a second scan,
e.g. for the outer `FROM sensors` in the ASOF-tolerance CTE query — DuckDB
re-acquires the GIL before calling the Python method):

```
hold GIL                  ← already held
  └─ reader()
       └─ inner.lock()    ← tries to acquire inner second
```

This is a textbook ABBA deadlock:

```
Scanner thread  : holds inner.lock()  → waits for GIL
Executor thread : holds GIL           → waits for inner.lock()
```

### Why only `test_asof_join_with_tolerance`?

The `_ASOF_TOLERANCE_SQL` query references `sensors` twice — once inside a
CTE and once in the outer `FROM` clause.  DuckDB materialises the CTE first
(Pipeline 1), then scans `sensors` again for the outer join (Pipeline 2).

When Pipeline 2's setup (which needs the GIL to call `__arrow_c_stream__`)
overlaps with Pipeline 1's scanner still reading from `sensors` (which holds
`inner.lock()` and waits for the GIL), the deadlock window opens.

Simpler queries — self-joins, single-table scans, LIMIT queries — either scan
each registered table only once or call `__arrow_c_stream__` before any
scanner threads start, so the window never opens.

The deadlock was non-deterministic (roughly 1-in-10 isolated runs) because it
depends on thread scheduling: Pipeline 2's `__arrow_c_stream__` call must
coincide with Pipeline 1's scanner being inside `ingest_up_to`.

### Why the GIL matters for the upstream reader

`PyRecordBatchReader::into_reader()` extracts a raw Arrow C stream from the
Python object.  The C stream's `get_next` function is pyarrow's
implementation, which calls Python C-API functions (`PyIter_Next`, etc.) and
**requires the GIL to be held**.  Calling it from a thread that does not hold
the GIL is undefined behaviour — it *usually* works (if no other Python thread
is active) but occasionally deadlocks or corrupts state.

`Python::allow_threads` is unavailable under `abi3` targets (pyo3 cannot
guarantee a GIL exists on Python 3.13+ no-GIL builds), so the raw FFI
functions `PyGILState_Ensure` / `PyGILState_Release` are used directly; both
are part of `Py_LIMITED_API` and are safe under abi3.

## The fix

### Invariant: always acquire `inner.lock()` before the GIL

The fix establishes a **strict, global lock-ordering rule**:

> `inner.lock()` is always acquired **before** the GIL.
> The GIL is **never** held while waiting for `inner.lock()`.

This makes the two orderings consistent, eliminating the ABBA pattern.

### Two helpers

```rust
/// Release the GIL, run `f`, re-acquire it.  Used in #[pymethods] to avoid
/// holding the GIL while blocking on inner.lock().
fn without_gil<T, F: FnOnce() -> T>(_py: Python<'_>, f: F) -> T { … }

/// Acquire the GIL, run `f`, release it back to its previous state.  Used
/// inside ingest_up_to (called with inner.lock() held) to safely call the
/// upstream Python reader from any thread, including DuckDB scanner threads.
fn with_gil_acquired<T, F: FnOnce() -> T>(f: F) -> T { … }
```

Both helpers use `PyEval_SaveThread` / `PyEval_RestoreThread` and
`PyGILState_Ensure` / `PyGILState_Release` from `Py_LIMITED_API`.  A
`#[cfg(not(Py_GIL_DISABLED))]` guard makes them no-ops on free-threaded
Python 3.13+.

### Call-site changes

**`ingest_up_to` (called with `inner.lock()` held)**
Wraps `upstream.next()` with `with_gil_acquired` so the upstream Python
reader is always called with the GIL regardless of which thread (DuckDB
scanner or Python caller) drives ingestion.

```
inner.lock() held
  └─ with_gil_acquired(|| upstream.next())   ← GIL acquired then released
```

**All `#[pymethods]` that access `inner`**
Now call `without_gil(py, || { inner.lock(); … })` so the GIL is dropped
before waiting for the mutex.  Affected methods: `reader()`, `__iter__()`,
`__arrow_c_stream__()`, `ingest_all()`, `ingested_count`, `upstream_exhausted`.

```
GIL held (pymethods entry)
  └─ without_gil()          ← GIL released
       └─ inner.lock()      ← acquired (no GIL)
            └─ work …
       └─ inner.unlock()
  └─ GIL re-acquired
```

### Why nested acquire/release is safe

`ingest_all` releases the GIL via `without_gil`, then `ingest_up_to` calls
`with_gil_acquired` (re-acquires the GIL) for each upstream batch.
`PyGILState_Ensure` / `PyGILState_Release` are designed for exactly this
pattern: `PyGILState_Release` restores the previous GIL state rather than
unconditionally releasing, so nesting is safe.

## Verification

The test `test_asof_join_with_tolerance` was run 50 consecutive times in
isolation (each run is a fresh Python process) and passed every time.  Before
the fix the hang rate was approximately 1-in-10.
