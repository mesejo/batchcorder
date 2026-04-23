"""Batchcorder: Replayable cached Arrow record-batch streams."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any, Self

from ._batchcorder import (
    CastingStreamCache as _PyCastingStreamCache,
)
from ._batchcorder import (
    StreamCache as _PyStreamCache,
)
from ._batchcorder import (
    StreamCacheReader as _PyStreamCacheReader,
)


__all__ = [
    "CastingStreamCache",
    "StreamCache",
    "StreamCacheReader",
]

__version__: str = version("batchcorder")


class StreamCache:
    """
    A cached Arrow stream backed by an in-memory Vec or an on-disk IPC file.

    Wraps any Arrow stream source and stores each ``RecordBatch`` so multiple
    independent :class:`StreamCacheReader` handles can replay the full stream
    from any position.  The upstream source is ingested lazily on demand and
    consumed exactly once.

    Two storage modes are supported:

    - **Memory-only** (omit ``disk_path`` / ``disk_capacity``): batches are
      kept as reference-counted pointers in RAM.  Reads are zero-copy; no IPC
      serialisation happens.
    - **Disk** (provide both ``disk_path`` and ``disk_capacity``): batches are
      serialised to an append-only Arrow IPC file.  A configurable hot layer
      (``memory_capacity``) keeps recently ingested batches in RAM to reduce
      disk reads.

    Prefer the named constructors :meth:`in_memory` and :meth:`on_disk` over
    calling this class directly.

    Parameters
    ----------
    reader : ArrowStreamExportable
        Any object implementing ``__arrow_c_stream__`` (e.g.
        :class:`pyarrow.Table`, :class:`pyarrow.RecordBatchReader`,
        :class:`arro3.core.RecordBatchReader`).
    memory_capacity : int, optional
        Hot-layer budget in bytes for disk mode.  Defaults to total physical
        RAM.  Ignored in memory-only mode.
    disk_path : str or Path, optional
        Directory for the on-disk IPC file.  Created on first use.
        Must be provided together with ``disk_capacity``.
    disk_capacity : int, optional
        On-disk storage budget in bytes.
        Must be provided together with ``disk_path``.

    Examples
    --------
    Memory-only:

    >>> import pyarrow as pa
    >>> from batchcorder import StreamCache
    >>> table = pa.table({"id": [1, 2, 3], "val": [0.5, 1.0, 1.5]})
    >>> ds = StreamCache.in_memory(table)
    >>> pa.RecordBatchReader.from_stream(ds).read_all().equals(table)
    True

    Disk mode:

    >>> import tempfile
    >>> tmp = tempfile.mkdtemp()
    >>> ds = StreamCache.on_disk(table, path=tmp, disk_capacity=64 << 20)
    >>> pa.RecordBatchReader.from_stream(ds).read_all().equals(table)
    True
    >>> ds.upstream_exhausted
    True

    As a context manager:

    >>> with StreamCache.on_disk(table, path=tmp, disk_capacity=64 << 20) as ds:
    ...     result = pa.RecordBatchReader.from_stream(ds).read_all()

    """

    def __init__(
        self,
        reader: Any,
        memory_capacity: int | None = None,
        disk_path: str | Path | None = None,
        disk_capacity: int | None = None,
    ):
        """See class docstring for parameter documentation."""
        self._impl = _PyStreamCache(
            reader,
            memory_capacity,
            str(disk_path) if disk_path is not None else None,
            disk_capacity,
        )

    # ── named constructors ────────────────────────────────────────────────────

    @classmethod
    def in_memory(cls, reader: Any, capacity: int | None = None) -> StreamCache:
        """
        Create a memory-only :class:`StreamCache`.

        Batches are stored as reference-counted pointers; reads are zero-copy.

        Parameters
        ----------
        reader : ArrowStreamExportable
            Any object implementing ``__arrow_c_stream__``.
        capacity : int, optional
            Memory budget in bytes.  Defaults to a fraction of system RAM.

        Returns
        -------
        StreamCache

        Examples
        --------
        >>> import pyarrow as pa
        >>> from batchcorder import StreamCache
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> ds = StreamCache.in_memory(table)
        >>> ds.ingest_all()
        1

        """
        return cls(reader, memory_capacity=capacity)

    @classmethod
    def on_disk(
        cls,
        reader: Any,
        path: str | Path,
        disk_capacity: int,
        memory_capacity: int | None = None,
    ) -> StreamCache:
        """
        Create a disk-backed :class:`StreamCache` with an optional hot layer.

        Batches are serialised to an append-only Arrow IPC file under *path*.
        Recently ingested batches are kept in a configurable hot layer in RAM
        to reduce disk reads.

        Parameters
        ----------
        reader : ArrowStreamExportable
            Any object implementing ``__arrow_c_stream__``.
        path : str or Path
            Directory for the on-disk IPC file.  Created on first use.
        disk_capacity : int
            On-disk storage budget in bytes.
        memory_capacity : int, optional
            Hot-layer budget in bytes.  Defaults to a fraction of system RAM.

        Returns
        -------
        StreamCache

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import StreamCache
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = StreamCache.on_disk(table, path=tmp, disk_capacity=64 << 20)
        >>> ds.ingest_all()
        1

        """
        return cls(
            reader,
            memory_capacity=memory_capacity,
            disk_path=path,
            disk_capacity=disk_capacity,
        )

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        """Enter the context manager, returning self."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the context manager, closing the cache."""
        self.close()

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def schema(self) -> Any:
        """
        Arrow schema of this dataset.

        Returns
        -------
        arro3.core.Schema

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import StreamCache
        >>> table = pa.table({"id": [1, 2], "val": [0.5, 1.0]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = StreamCache.on_disk(table, tmp, 64 << 20)
        >>> [f.name for f in ds.schema]
        ['id', 'val']

        """
        return self._impl.schema

    @property
    def ingested_count(self) -> int:
        """
        Number of batches pulled from the upstream source so far.

        Increments lazily as readers consume batches.

        Returns
        -------
        int

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import StreamCache
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = StreamCache.on_disk(table, tmp, 64 << 20)
        >>> ds.ingested_count
        0
        >>> ds.ingest_all()
        1
        >>> ds.ingested_count
        1

        """
        return self._impl.ingested_count

    @property
    def upstream_exhausted(self) -> bool:
        """
        ``True`` once the upstream source has been fully consumed.

        Returns
        -------
        bool

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import StreamCache
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = StreamCache.on_disk(table, tmp, 64 << 20)
        >>> ds.upstream_exhausted
        False
        >>> ds.ingest_all()
        1
        >>> ds.upstream_exhausted
        True

        """
        return self._impl.upstream_exhausted

    # ── dunder methods ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a string representation of the cache."""
        return (
            f"StreamCache("
            f"ingested={self.ingested_count}, "
            f"exhausted={self.upstream_exhausted}, "
            f"schema={self.schema}"
            f")"
        )

    def __len__(self) -> int:
        """
        Return the number of ingested batches.

        Raises
        ------
        TypeError
            If the upstream source has not been fully consumed yet.  Call
            :meth:`ingest_all` first, or check :attr:`upstream_exhausted`.

        """
        if not self.upstream_exhausted:
            raise TypeError(
                "len() is not available until the stream is fully ingested; "
                "call ingest_all() first or check upstream_exhausted"
            )
        return self.ingested_count

    def __iter__(self) -> StreamCacheReader:
        """
        Iterate over all batches from the start.

        Creates a fresh :class:`StreamCacheReader` starting at batch 0 and
        returns it as the iterator.

        Returns
        -------
        StreamCacheReader

        """
        return self.reader(True)

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """
        Enable Arrow stream export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

        This dunder method should not be called directly, but enables zero-copy data
        transfer to other Python libraries that understand Arrow memory.

        Creates a fresh reader starting at batch 0.  Allows the dataset to be
        consumed directly by PyArrow, DuckDB, DataFusion, and any other
        Arrow-compatible library.

        Parameters
        ----------
        requested_schema : object, optional
            Schema capsule to cast the stream to, or ``None``.

        """
        return self._impl.__arrow_c_stream__(requested_schema)

    def __arrow_c_schema__(self) -> Any:
        """
        Enable Arrow schema export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

        This dunder method should not be called directly, but enables zero-copy data
        transfer to other Python libraries that understand Arrow memory.

        This allows Arrow consumers to inspect the data type of this
        :class:`StreamCache`.  Then the consumer can ask the producer (in
        ``__arrow_c_stream__``) to cast the exported data to a supported data type.

        """
        return self._impl.__arrow_c_schema__()

    # ── methods ───────────────────────────────────────────────────────────────

    def reader(self, from_start: bool = True) -> StreamCacheReader:
        """
        Return a new :class:`StreamCacheReader` handle.

        Parameters
        ----------
        from_start : bool, optional
            If ``True`` (default), the reader starts at batch 0 and replays the
            full stream.  If ``False``, it starts at the current ingestion
            frontier and yields only batches ingested after this call.

        Returns
        -------
        StreamCacheReader

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import StreamCache
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = StreamCache.on_disk(table, tmp, 64 << 20)
        >>> r1 = ds.reader()
        >>> r2 = ds.reader()
        >>> r1.closed, r2.closed
        (False, False)

        """
        return StreamCacheReader(self._impl.reader(from_start))

    def cast(self, target_schema: Any) -> CastingStreamCache:
        """
        Cast the dataset to produce batches with the given schema.

        Returns a :class:`CastingStreamCache` — a **replayable** wrapper that
        applies the schema cast on every read.  Unlike
        :meth:`pyarrow.RecordBatchReader.cast`, the result can be consumed
        multiple times, making it suitable for DuckDB self-joins and ASOF joins.

        Parameters
        ----------
        target_schema : ArrowSchemaExportable
            Any Arrow schema-compatible object (e.g. :class:`pyarrow.Schema`,
            :class:`arro3.core.Schema`).

        Returns
        -------
        CastingStreamCache

        """
        return CastingStreamCache(self._impl.cast(target_schema))

    def ingest_all(self) -> int:
        """
        Eagerly ingest all batches from the upstream source into the cache.

        After this call ``upstream_exhausted`` is ``True`` and the upstream
        reference is released.  Subsequent reads are served entirely from cache.
        Calling this method more than once is safe and idempotent.

        Returns
        -------
        int
            Total number of batches ingested (including any ingested previously).

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import StreamCache
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = StreamCache.on_disk(table, tmp, 64 << 20)
        >>> ds.ingest_all()
        1
        >>> ds.upstream_exhausted
        True

        """
        return self._impl.ingest_all()

    def close(self) -> None:
        """
        Close the dataset and destroy the underlying storage.

        This method clears the hybrid cache and destroys the disk storage,
        removing any unused files that were eagerly created.

        Returns
        -------
        None

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import StreamCache
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = StreamCache.on_disk(table, tmp, 64 << 20)
        >>> ds.close()

        """
        return self._impl.close()


class StreamCacheReader:
    """
    A single-use iterator handle for a :class:`StreamCache`.

    Maintains an independent read position.  Multiple handles backed by the
    same dataset share the underlying cache; the upstream source is ingested
    lazily as needed.

    Once consumed via ``__arrow_c_stream__`` or by exhausting iteration the
    reader is marked closed and raises an error on further use.

    Notes
    -----
    Obtain a handle from :meth:`StreamCache.reader` rather than constructing
    one directly.

    """

    def __init__(self, impl: _PyStreamCacheReader):
        """Obtain via :meth:`StreamCache.reader`."""
        self._impl = impl

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        """Enter the context manager, returning self."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the context manager; reader is consumed by iteration."""
        pass

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def schema(self) -> Any:
        """
        Arrow schema of batches produced by this reader.

        Returns
        -------
        arro3.core.Schema

        Raises
        ------
        ValueError
            If the reader has already been consumed.

        """
        return self._impl.schema

    @property
    def closed(self) -> bool:
        """
        ``True`` if this reader has been consumed.

        Returns
        -------
        bool

        """
        return self._impl.closed

    # ── dunder methods ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a string representation of the reader."""
        return f"StreamCacheReader(closed={self.closed}, schema={self._impl.schema})"

    def __iter__(self) -> StreamCacheReader:
        """Return self as the iterator."""
        return self

    def __next__(self) -> Any:
        """Get the next batch from the reader."""
        return next(self._impl)

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """
        Enable Arrow stream export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

        This dunder method should not be called directly, but enables zero-copy data
        transfer to other Python libraries that understand Arrow memory.

        Consumes the reader; subsequent calls raise an error.

        Parameters
        ----------
        requested_schema : object, optional
            Schema capsule to cast the stream to, or ``None``.

        Raises
        ------
        ValueError
            If the reader has already been consumed.

        """
        return self._impl.__arrow_c_stream__(requested_schema)

    def __arrow_c_schema__(self) -> Any:
        """
        Enable Arrow schema export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

        This dunder method should not be called directly, but enables zero-copy data
        transfer to other Python libraries that understand Arrow memory.

        This allows Arrow consumers to inspect the data type of this
        :class:`StreamCacheReader`.  Then the consumer can ask the producer (in
        ``__arrow_c_stream__``) to cast the exported data to a supported data type.

        Raises
        ------
        ValueError
            If the reader has already been consumed.

        """
        return self._impl.__arrow_c_schema__()

    # ── methods ───────────────────────────────────────────────────────────────

    def cast(self, target_schema: Any) -> Any:
        """
        Cast the reader to produce batches with the given schema.

        Mirrors :meth:`pyarrow.RecordBatchReader.cast`.  Returns a
        :class:`pyarrow.RecordBatchReader` that applies the cast as batches are
        read.  Consumes this reader.

        Parameters
        ----------
        target_schema : ArrowSchemaExportable
            Any Arrow schema-compatible object (e.g. :class:`pyarrow.Schema`,
            :class:`arro3.core.Schema`).

        Returns
        -------
        pyarrow.RecordBatchReader

        Raises
        ------
        ValueError
            If the reader has already been consumed.

        """
        return self._impl.cast(target_schema)


class CastingStreamCache:
    """
    A replayable cast view of a :class:`StreamCache`.

    Created by :meth:`StreamCache.cast`.  Each call to ``__arrow_c_stream__``
    produces a fresh reader from the underlying cache with each batch cast to
    :attr:`schema`, so this object is **replayable** — DuckDB self-joins, ASOF
    joins, and other multi-scan consumers work correctly on it.

    Notes
    -----
    Obtain via :meth:`StreamCache.cast` rather than constructing directly.

    """

    def __init__(self, impl: _PyCastingStreamCache):
        """Obtain via :meth:`StreamCache.cast`."""
        self._impl = impl

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def schema(self) -> Any:
        """
        Arrow schema produced by this dataset after casting.

        Returns
        -------
        arro3.core.Schema

        """
        return self._impl.schema

    # ── dunder methods ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a string representation of the casting cache."""
        return f"CastingStreamCache(schema={self.schema})"

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """
        Enable Arrow stream export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

        Creates a fresh reader from the underlying cache and applies the cast.
        Safe to call multiple times — each call produces an independent stream.

        Parameters
        ----------
        requested_schema : object, optional
            Schema capsule to further cast the stream to, or ``None`` (uses
            :attr:`schema`).

        """
        return self._impl.__arrow_c_stream__(requested_schema)

    def __arrow_c_schema__(self) -> Any:
        """
        Enable Arrow schema export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

        Returns the target schema so consumers can inspect the post-cast type.

        """
        return self._impl.__arrow_c_schema__()

    # ── methods ───────────────────────────────────────────────────────────────

    def cast(self, target_schema: Any) -> CastingStreamCache:
        """
        Cast to a further target schema, returning a new :class:`CastingStreamCache`.

        Parameters
        ----------
        target_schema : ArrowSchemaExportable
            Any Arrow schema-compatible object.

        Returns
        -------
        CastingStreamCache

        """
        return CastingStreamCache(self._impl.cast(target_schema))
