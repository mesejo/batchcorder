# batchcorder-datafusion

A DataFusion table provider that tees one read-once Arrow stream to every
scan in a query.

Register a `TeeTableProvider` with a DataFusion `SessionContext`, and every
scan of the table in a query (self-joins, CTEs referenced twice, subqueries)
replays the same upstream stream, which is read only once. With the
default `bounding="sealed"`, each planned scan reserves a reader at planning
time, and each batch is evicted as soon as every scan has read past it.

## Install

The provider plugs into a DataFusion host through `datafusion-ffi`. Pick the
extra for the host you use:

```bash
pip install "batchcorder-datafusion[datafusion]"   # datafusion-python
pip install "batchcorder-datafusion[xorq]"         # xorq-datafusion
```

Without an extra, no host is installed; bring your own, within the ranges
below.

## Usage

```python
import pyarrow as pa
from datafusion import SessionContext

from batchcorder_datafusion import TeeTableProvider


schema = pa.schema([("k", pa.int64())])
reader = pa.RecordBatchReader.from_batches(
    schema, [pa.record_batch([[1, 2, 3]], schema=schema)]
)

ctx = SessionContext()
ctx.register_table("t", TeeTableProvider(reader))
ctx.sql("select * from t a join t b using (k)").collect()
```

The upstream can be any object that implements the Arrow PyCapsule stream
interface (`__arrow_c_stream__`).

A sealed provider serves one query. Register a new provider per query, or
pass `bounding="unbounded"` for a table that is queried repeatedly.

### Do not pass a `batchcorder.StreamCache`

The provider keeps its own cache. Passing a `batchcorder` reader caches the
data twice. Pass the original upstream reader, unless you also need to replay
the stream outside DataFusion.

## Compatibility

`datafusion-ffi` structs are only compatible within one DataFusion major, so
the provider and the host must be built on the same one. The major version of
`batchcorder-datafusion` is that DataFusion major.

| batchcorder-datafusion | DataFusion | datafusion (datafusion-python) | xorq-datafusion |
|------------------------|------------|--------------------------------|-----------------|
| 55.x                   | 55         | `>=55,<56`                     | `>=0.3,<0.4`    |

Importing `batchcorder_datafusion` emits a `RuntimeWarning` for each installed
host outside these ranges. Registering the provider with such a host can fail
or crash the process.

`batchcorder-datafusion` does not depend on the `batchcorder` package. The
cache is compiled into its extension, and data enters through the Arrow C
stream interface.
