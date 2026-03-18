# Handle eviction errors

Foyer evicts cache entries under memory and disk pressure. If a reader tries to
fetch a batch that has been evicted from both tiers, it raises an `IOError`:

```
Batch N was evicted from the cache before it could be read.
Increase cache capacity so it can hold all live consumer positions.
```

## Why eviction happens

Eviction occurs when the total size of cached batches exceeds `disk_capacity`.
This is most likely when:

- Many concurrent readers are spread far apart (slow reader lags far behind a
  fast reader).
- The dataset is large relative to the configured disk capacity.
- `disk_capacity` was set too small.

## Prevention

Size the disk tier to hold at least the span of batches between your slowest
and fastest concurrent reader at any moment:

```python
# Example: 10 readers, each potentially 100 batches apart,
# each batch ~1 MB → need at least 100 MB disk headroom
ds = CachedDataset(
    reader,
    memory_capacity=256 << 20,
    disk_path="/tmp/cache",
    disk_capacity=2 * 1024 << 20,  # generous margin
)
```

## Detecting eviction in code

```python
import batchcorder

try:
    for batch in ds:
        process(batch)
except IOError as e:
    if "evicted" in str(e):
        # increase capacity and retry from a fresh CachedDataset
        ...
    raise
```

## Recovery

There is no way to recover a batch once it has been evicted. You must create a
new `CachedDataset` from the upstream source with a larger capacity. If the
upstream source is truly single-use and has already been exhausted, you cannot
recover — size the cache correctly up front.
