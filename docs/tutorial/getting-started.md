# Getting started

This tutorial walks you through installing batchcorder and caching your first
Arrow stream. By the end you will have a `CachedDataset` that you can read
multiple times from a source that can only be consumed once.

## Prerequisites

- Python 3.11 or later
- A Rust toolchain ([rustup.rs](https://rustup.rs))
- [uv](https://docs.astral.sh/uv/)

## Install

Clone the repository and build the extension in development mode:

```bash
git clone https://github.com/yourorg/multirecord
cd multirecord
uv sync
maturin develop --uv
```

## Create your first CachedDataset

```python
import tempfile
import pyarrow as pa
from batchcorder import CachedDataset

# Any Arrow-compatible source works here.
# We use a simple in-memory table split into one-row batches.
table = pa.table({"id": [1, 2, 3], "value": [0.5, 1.0, 1.5]})
reader = table.to_reader(max_chunksize=1)

cache_dir = tempfile.mkdtemp()

ds = CachedDataset(
    reader,
    memory_capacity=64 * 1024 * 1024,   # 64 MB in-memory tier
    disk_path=cache_dir,
    disk_capacity=512 * 1024 * 1024,    # 512 MB on-disk tier
)
```

The upstream `reader` is not consumed yet — ingestion is lazy.

## Read it back

```python
# First pass: upstream is consumed as batches are fetched
result1 = pa.RecordBatchReader.from_stream(ds).read_all()

# Second pass: served entirely from cache, upstream already exhausted
result2 = pa.RecordBatchReader.from_stream(ds).read_all()

assert result1.equals(result2)
print(ds.upstream_exhausted)  # True
```

## Pre-ingest upfront

If you want to consume the upstream source immediately before any reader
touches the data, call `ingest_all()`:

```python
ds.ingest_all()
# upstream is now fully cached; safe to discard the original source
```

## Get independent reader handles

Rather than using `__arrow_c_stream__` directly, you can request explicit
`CachedDatasetReader` handles that each maintain their own position:

```python
r1 = ds.reader()          # starts at batch 0
r2 = ds.reader()          # also starts at batch 0, independent of r1

batch_a = next(r1)
batch_b = next(r1)        # r1 is now at batch 2
batch_c = next(r2)        # r2 is still at batch 0
```

## Next steps

- [Use with DuckDB](../how-to/duckdb.md) — register a dataset as a DuckDB
  virtual table for SQL queries.
- [Configure cache size](../how-to/cache-config.md) — choose memory and disk
  capacities for your workload.
- [Handle eviction errors](../how-to/eviction.md) — what to do when a batch
  is evicted before a reader reaches it.
