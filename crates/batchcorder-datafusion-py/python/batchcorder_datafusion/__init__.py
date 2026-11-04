"""
DataFusion table provider that tees one read-once Arrow stream to every scan.

Register a :class:`TeeTableProvider` with a datafusion-python
``SessionContext`` and every scan of the table in a query replays the same
upstream stream, which is read only once.  With the default
``bounding="sealed"``, each planned scan reserves a reader at planning time,
and each batch is evicted as soon as every scan has read past it.

Example::

    import pyarrow as pa
    from datafusion import SessionContext
    from batchcorder_datafusion import TeeTableProvider

    ctx = SessionContext()
    ctx.register_table("t", TeeTableProvider(reader))
    ctx.sql("select * from t a join t b using (k)").collect()

A sealed provider serves one query.  Register a new provider per query, or
pass ``bounding="unbounded"`` for a table that is queried repeatedly.
"""

from __future__ import annotations

from importlib.metadata import version

from ._batchcorder_datafusion import TeeTableProvider


__all__ = ["TeeTableProvider"]

__version__: str = version("batchcorder-datafusion")
