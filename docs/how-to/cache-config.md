# Configure cache size

`CachedDataset` takes four constructor arguments that control how batches are
stored. Choosing the right sizes avoids both wasted memory and eviction errors.

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `memory_capacity` | `int` (bytes) | Maximum bytes held in the RAM tier |
| `disk_path` | `str` | Directory for the on-disk tier |
| `disk_capacity` | `int` (bytes) | Maximum bytes held on disk |

```python
ds = CachedDataset(
    reader,
    memory_capacity=256 * 1024 * 1024,   # 256 MB RAM
    disk_path="/data/cache/batchcorder",
    disk_capacity=4 * 1024 * 1024 * 1024, # 4 GB disk
)
```

## Sizing rules of thumb

**Memory tier** — hold the batches that your fastest reader is ahead of your
slowest reader. If all readers advance at the same pace, a few hundred MB is
usually enough.

**Disk tier** — hold the entire dataset if you cannot guarantee all readers
will finish before entries age out of the memory tier.

## Eviction

When the memory tier is full, Foyer evicts entries to disk in the background.
When the disk tier is full, entries are dropped entirely. A reader that
requests a fully-evicted batch raises an error.

See [Handle eviction errors](eviction.md) for how to detect and recover from
this.

## Estimating batch size

```python
import pyarrow as pa

batch = next(iter(reader))
ipc_size = len(batch.serialize())   # approximate bytes in cache
print(f"~{ipc_size / 1024:.1f} KiB per batch")
```

The cache stores batches serialized as Arrow IPC, so `batch.serialize()` gives
a close estimate of the per-batch footprint.
