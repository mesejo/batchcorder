# Disk-Spill Testing for CachedDataset

## Background

`CachedDataset` stores each `RecordBatch` in a Foyer hybrid cache keyed by a
monotonic batch index.  Foyer has two tiers:

| Tier | Speed | Capacity |
|------|-------|----------|
| Memory | Fast (RAM) | Fixed at construction — `memory_capacity` bytes |
| Disk | Slower (filesystem) | Fixed at construction — `disk_capacity` bytes |

When the memory tier is full, Foyer **evicts** entries to the disk tier in the
background.  The critical guarantee that `CachedDataset` relies on is that
`cache.get(idx)` will find the entry on disk even after it has left RAM — as
long as the disk tier has not also been exhausted.

The default test fixtures in `test_cached_dataset.py` use a 16 MiB memory
tier, which comfortably holds the tiny test tables and never triggers a spill.
The disk-spill tests exercise the code path where **every batch must be
fetched from the disk tier**, ensuring the full memory→disk→reader pipeline is
correct.

---

## How Spill Is Forced

Foyer evicts a cache entry from the memory tier as soon as the total size of
in-memory entries exceeds `memory_capacity`.  To guarantee that every batch
spills, the test fixtures choose:

```
memory_capacity  <<  serialized_size_of_one_batch
```

The concrete choice used throughout is:

```python
_SPILL_MEMORY = 32 * 1024  # 32 KiB memory tier
```

combined with batches that carry a binary `pad` column:

```python
"pad": [b"x" * 1024 * 50] * batch.num_rows  # 50 KiB per row
```

A three-row batch therefore serializes to roughly **150 KiB** (50 KiB × 3
rows, plus Arrow IPC framing), far above the 32 KiB ceiling.  After the first
`cache.insert`, the memory tier is over capacity and Foyer schedules the entry
for disk persistence.  By the time any reader calls `cache.get`, the entry is
on disk.

### Why a `pad` column instead of a tiny `memory_capacity`

Setting `memory_capacity=1` would also force spill, but Foyer's internal
bookkeeping may behave unexpectedly with sub-kilobyte capacities.  Using a
large payload with a comfortable 32 KiB tier is explicit, predictable, and
matches the pattern established in `test_cached_dataset.py`.

---

## Tests Added to `test_cached_dataset.py`

### `test_disk_spill_and_pyarrow_ipc_roundtrip`

Verifies the end-to-end pipeline when the memory tier is smaller than a
single batch:

1. Builds a table with 5 batches × 100 rows × 1 KiB binary payload (≈ 100 KiB
   per batch).
2. Constructs a `CachedDataset` with a 32 KiB memory tier.
3. Calls `ingest_all()` — every batch is pushed through the cache and evicted
   to disk.
4. Asserts that `tmp_path` contains at least one file (Foyer's block device
   files, created at construction and populated on eviction).
5. Reads back via `pa.RecordBatchReader.from_stream(ds)`, which calls
   `__arrow_c_stream__`, fetches IPC bytes from the disk tier, and decodes
   them.
6. Asserts `result.equals(table)` — the round-trip is lossless.

### `test_disk_spill_ipc_file_write_and_read`

Extends the spill round-trip to exercise PyArrow's IPC **file** format
(random-access) in addition to the C stream protocol:

1. Same spill setup as above, with separate subdirectories for the Foyer cache
   (`cache_dir`) and the IPC output (`ipc_path`).
2. After `ingest_all()`, asserts Foyer files exist under `cache_dir`.
3. Streams the dataset out via `pa.RecordBatchReader.from_stream(ds)` and
   writes each batch to an Arrow IPC file using `pa.ipc.new_file()`.
4. Reads the IPC file back with `pa.memory_map` + `pa.ipc.open_file()` — the
   zero-copy, random-access path described in the Arrow documentation.
5. Asserts the loaded table equals the original.

This test confirms that data surviving a Foyer disk eviction is also
representable in the standard Arrow IPC file format and readable with PyArrow's
memory-mapped reader.

```
upstream
  └─ CachedDataset (32 KiB memory, disk spill)
       └─ pa.RecordBatchReader.from_stream()   ← Arrow C stream from disk tier
            └─ pa.ipc.new_file() writer        ← Arrow IPC file format
                 └─ pa.memory_map + open_file  ← zero-copy read-back
```

---

## Tests Added to `test_duckdb.py`

All three DuckDB spill tests share the same strategy: pad existing test
fixtures with a large binary column so that each batch exceeds the 32 KiB
memory tier, then reuse the existing SQL and reference-result helpers
unchanged, since the queries never project the `pad` column.

### Fixtures

**`_PADDED_EMPLOYEE_BATCHES` / `_PADDED_EMPLOYEE_SCHEMA`**

Derives from `EMPLOYEE_BATCHES` by adding `"pad": [b"x" * 1024 * 50] *
batch.num_rows`.  Each 3-row batch becomes ≈ 150 KiB — 4.5× the memory tier.

**`_PADDED_SENSOR_BATCHES` / `_PADDED_SENSOR_BATCHES`**

Derives from `_SENSOR_BATCHES` and `_EVENT_BATCHES` respectively using the
same 50 KiB per-row pad.

**`_spill_agg_batches()`**

Ten batches of 100 rows with a 1 KiB pad per row (≈ 100 KiB per batch), used
for the aggregation test.

### `test_self_join_with_disk_spill`

Mirrors `test_self_join` but with the padded employee fixtures and a 32 KiB
memory tier.

DuckDB scans `employees` twice — once per side of the hash join.
`CachedDataset` serves the second scan from the disk tier.  Assertions:

- Join result equals the known employee hierarchy.
- `source.batches_read == 2` — upstream was ingested exactly once.
- `ds.upstream_exhausted is True`.
- Foyer disk files exist under `tmp_path`.

### `test_aggregation_with_disk_spill`

Runs four aggregation queries — `COUNT`, `SUM`, `AVG`, `MIN/MAX` — against
`_spill_agg_batches()` stored in a 32 KiB memory tier.  Each query result is
compared against `_spill_agg_reference(sql)`, which runs the same SQL on an
in-memory PyArrow table (the `_reference_result` pattern from the existing
ASOF-join tests).

This is a sequential single-scan workload; its purpose is to confirm that
DuckDB can compute correct aggregates when every batch must be fetched from
disk.

### `test_asof_join_with_tolerance_and_disk_spill`

Mirrors `test_asof_join_with_tolerance` but with padded sensor/event fixtures
and a 32 KiB memory tier for both datasets.

This is the most demanding spill test because `_ASOF_TOLERANCE_SQL` is a
two-stage query:

```sql
WITH matched AS (
    SELECT s.site, s.event_time, e.event_type
    FROM sensors s
    ASOF LEFT JOIN events e
        ON s.site = e.site AND s.event_time >= e.event_time
    WHERE e.event_time IS NULL
       OR s.event_time - e.event_time <= INTERVAL 1 SECOND
)
SELECT s.site, s.humidity, m.event_type
FROM sensors s
LEFT JOIN matched m ON s.site = m.site AND s.event_time = m.event_time
ORDER BY s.event_time
```

DuckDB scans `sensors` twice — once inside the CTE and once for the outer
`FROM sensors`.  Both scans must be served from the disk tier.  The GIL /
Mutex ordering fix documented in [GIL & locking design](gil-deadlock.md) is
what makes this safe.

Assertions:

- Query result equals `_reference_result(_ASOF_TOLERANCE_SQL)` — the same
  ground-truth helper used by the non-spill variant.
- `sensors_ds.upstream_exhausted is True` and `events_ds.upstream_exhausted is True`.
- Both `tmp_path/sensors` and `tmp_path/events` contain Foyer block files.

---

## Key Observations

### Foyer disk files appear at construction, not on first write

`FsDeviceBuilder::build()` creates and initialises the on-disk block device
during `HybridCacheBuilder::build().await`.  The `tmp_path.rglob("*")` check
therefore passes even before any batch has been inserted; its real purpose is
to confirm that Foyer's disk engine was not silently disabled or redirected.

### Disk-tier fetches happen after the Mutex is released

The `CachedDatasetReaderImpl::next()` implementation deliberately releases
`inner.lock()` before calling `cache.get(&idx).await`:

```
Step 1 — lock(inner), call ingest_up_to, update consumer_positions, clone cache handle
Step 2 — unlock(inner)
Step 3 — cache.get(&idx).await  ← disk I/O here, no lock held
Step 4 — deserialize IPC bytes
```

This means multiple readers can fetch different batches from disk
concurrently, even though ingestion is serialised by the Mutex.

### Eviction between ingest and fetch

If a batch is evicted from **both** tiers between Steps 1 and 3 (because the
disk tier is also full), `cache.get` returns `None` and the reader raises:

```
Batch N was evicted from the cache before it could be read.
Increase cache capacity so it can hold all live consumer positions.
```

The spill tests avoid this by setting `disk_capacity` large enough to hold the
entire dataset many times over.

### The `pad` column is invisible to SQL

All DuckDB queries project only the original columns (`site`, `humidity`,
`event_type`, `id`, `value`).  The `pad` column is present in the Arrow schema
seen by DuckDB but is never referenced, so query results are identical to the
non-padded reference runs.  This lets the existing `_reference_result` helper
serve as ground truth without modification.
