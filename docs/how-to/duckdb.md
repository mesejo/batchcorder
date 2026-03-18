# Use with DuckDB

`CachedDataset` and `CachedDatasetReader` implement the
[Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
(`__arrow_c_stream__` / `__arrow_c_schema__`), so DuckDB can register them
directly as virtual tables.

## Basic query

```python
import tempfile
import pyarrow as pa
import duckdb
from batchcorder import CachedDataset

table = pa.table({
    "id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Charlie", "Alice"],
    "value": [10.0, 20.0, 30.0, 40.0],
})

ds = CachedDataset(
    table,
    memory_capacity=64 << 20,
    disk_path=tempfile.mkdtemp(),
    disk_capacity=512 << 20,
)

result = duckdb.sql("SELECT name, SUM(value) FROM ds GROUP BY name").fetchdf()
print(result)
```

## Self-join

Because `CachedDataset` serves each new `__arrow_c_stream__` call from the
cache, DuckDB can scan the same dataset multiple times in a single query
without exhausting the upstream source:

```python
result = duckdb.sql("""
    SELECT a.id, a.name, b.name AS manager
    FROM ds a
    LEFT JOIN ds b ON a.id = b.id + 1
""").fetchdf()
```

## ASOF join across two datasets

```python
sensors_ds = CachedDataset(sensors_table, memory_capacity=64 << 20,
                            disk_path=tempfile.mkdtemp(), disk_capacity=512 << 20)
events_ds  = CachedDataset(events_table,  memory_capacity=64 << 20,
                            disk_path=tempfile.mkdtemp(), disk_capacity=512 << 20)

result = duckdb.sql("""
    SELECT s.site, s.humidity, e.event_type
    FROM sensors_ds s
    ASOF LEFT JOIN events_ds e
        ON s.site = e.site AND s.event_time >= e.event_time
""").fetchdf()
```

!!! note "GIL / locking"
    DuckDB drives scans from its own C++ threads. batchcorder handles this
    safely via a strict lock-ordering invariant between its internal mutex and
    the Python GIL. See [GIL & locking design](../explanation/gil-deadlock.md)
    for details.
