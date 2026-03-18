# batchcorder

A Rust-backed Python library for caching Arrow record-batch streams so they can
be replayed multiple times from a source that can only be read once.

## The problem

Arrow `RecordBatchReader` is a single-use stream — once consumed, it is gone.
Training loops and multi-pass data pipelines need to iterate the same dataset
repeatedly without re-reading from disk or the network each time.

## What batchcorder does

`CachedDataset` wraps any Arrow stream source (anything that implements
`__arrow_c_stream__`) and stores each `RecordBatch` in a two-tier hybrid cache
(memory + disk) backed by [Foyer](https://github.com/foyer-rs/foyer).
Multiple independent readers can then replay the stream concurrently, each
maintaining their own position in the batch sequence.

```
upstream source  →  CachedDataset  →  CachedDatasetReader 0  (from batch 0)
  (read once)      [mem + disk]    →  CachedDatasetReader 1  (from batch 0)
                                   →  CachedDatasetReader 2  (from batch 3)
```

## Key properties

- **Single-read source**: the upstream stream is consumed exactly once; all
  subsequent reads come from the cache.
- **Concurrent readers**: multiple `CachedDatasetReader` instances from the
  same dataset are fully independent and thread-safe.
- **Lazy ingestion**: batches are fetched from the upstream source on demand as
  readers advance, not upfront.
- **Replay from any position**: `ds.reader(from_start=True)` replays from
  batch 0; `ds.reader(from_start=False)` starts from the current frontier.

## Where to go next

New to batchcorder? Start with the [tutorial](tutorial/getting-started.md).

Looking for a specific task? See the [how-to guides](how-to/duckdb.md).

Need the full API? See the [reference](reference/cached-dataset.md).

Want to understand the design? Read the [explanation](explanation/architecture.md).
