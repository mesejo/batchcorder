"""Batchcorder: Hybrid memory+disk cached Arrow datasets."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any

from ._batchcorder import (
    CachedDataset as _PyCachedDataset,
)
from ._batchcorder import (
    CachedDatasetReader as _PyCachedDatasetReader,
)
from ._batchcorder import (
    CastingDataset as _PyCastingDataset,
)


__all__ = [
    "CachedDataset",
    "CachedDatasetReader",
    "CastingDataset",
]

__version__: str = version("batchcorder")


class CachedDataset:
    """A hybrid memory+disk cached Arrow dataset.

    Wraps any Arrow stream source and caches each ``RecordBatch`` in a Foyer
    hybrid cache keyed by a monotonic batch index.  Multiple independent
    :class:`CachedDatasetReader` handles can replay the full stream from any
    position; the upstream source is ingested lazily on demand.

    Parameters
    ----------
    reader : object
        Any object implementing ``__arrow_c_stream__`` (e.g.
        :class:`pyarrow.Table`, :class:`pyarrow.RecordBatchReader`,
        :class:`arro3.core.RecordBatchReader`).
    memory_capacity : int
        In-memory cache tier size in bytes.
    disk_path : str
        Directory for the on-disk cache tier.  Created on first use.
    disk_capacity : int
        On-disk cache tier size in bytes.

    Notes
    -----
    Foyer may evict cache entries under memory/disk pressure.  If a batch is
    evicted before a reader reaches it the reader raises an error.  Size both
    tiers so they can hold all live reader positions simultaneously.

    Examples
    --------
    >>> import tempfile
    >>> import pyarrow as pa
    >>> from batchcorder import CachedDataset
    >>> table = pa.table({"id": [1, 2, 3], "val": [0.5, 1.0, 1.5]})
    >>> tmp = tempfile.mkdtemp()
    >>> ds = CachedDataset(table, memory_capacity=16 << 20, disk_path=tmp, disk_capacity=64 << 20)
    >>> pa.RecordBatchReader.from_stream(ds).read_all().equals(table)
    True
    >>> ds.upstream_exhausted
    True

    """

    def __init__(
        self, reader: Any, memory_capacity: int, disk_path: str, disk_capacity: int
    ):
        """See class docstring for parameter documentation."""
        self._impl = _PyCachedDataset(reader, memory_capacity, disk_path, disk_capacity)

    @property
    def schema(self) -> Any:
        """Arrow schema of this dataset.

        Returns
        -------
        arro3.core.Schema

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import CachedDataset
        >>> table = pa.table({"id": [1, 2], "val": [0.5, 1.0]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = CachedDataset(table, 16 << 20, tmp, 64 << 20)
        >>> [f.name for f in ds.schema]
        ['id', 'val']

        """
        return self._impl.schema

    @property
    def ingested_count(self) -> int:
        """Number of batches pulled from the upstream source so far.

        Increments lazily as readers consume batches.

        Returns
        -------
        int

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import CachedDataset
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = CachedDataset(table, 16 << 20, tmp, 64 << 20)
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
        """``True`` once the upstream source has been fully consumed.

        Returns
        -------
        bool

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import CachedDataset
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = CachedDataset(table, 16 << 20, tmp, 64 << 20)
        >>> ds.upstream_exhausted
        False
        >>> ds.ingest_all()
        1
        >>> ds.upstream_exhausted
        True

        """
        return self._impl.upstream_exhausted

    def reader(self, from_start: bool = True) -> CachedDatasetReader:
        """Return a new :class:`CachedDatasetReader` handle.

        Parameters
        ----------
        from_start : bool, optional
            If ``True`` (default), the reader starts at batch 0 and replays the
            full stream.  If ``False``, it starts at the current ingestion
            frontier and yields only batches ingested after this call.

        Returns
        -------
        CachedDatasetReader

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import CachedDataset
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = CachedDataset(table, 16 << 20, tmp, 64 << 20)
        >>> r1 = ds.reader()
        >>> r2 = ds.reader()
        >>> r1.closed, r2.closed
        (False, False)

        """
        return CachedDatasetReader(self._impl.reader(from_start))

    def __iter__(self) -> CachedDatasetReader:
        """Iterate over all batches from the start.

        Creates a fresh :class:`CachedDatasetReader` starting at batch 0 and
        returns it as the iterator.

        Returns
        -------
        CachedDatasetReader

        """
        return self.reader(True)

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """Enable Arrow stream export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

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
        """Enable Arrow schema export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

        This dunder method should not be called directly, but enables zero-copy data
        transfer to other Python libraries that understand Arrow memory.

        This allows Arrow consumers to inspect the data type of this
        :class:`CachedDataset`.  Then the consumer can ask the producer (in
        ``__arrow_c_stream__``) to cast the exported data to a supported data type.

        """
        return self._impl.__arrow_c_schema__()

    def cast(self, target_schema: Any) -> CastingDataset:
        """Cast the dataset to produce batches with the given schema.

        Returns a :class:`CastingDataset` — a **replayable** wrapper that
        applies the schema cast on every read.  Unlike
        :meth:`pyarrow.RecordBatchReader.cast`, the result can be consumed
        multiple times, making it suitable for DuckDB self-joins and ASOF joins.

        Parameters
        ----------
        target_schema : object
            Any Arrow schema-compatible object (e.g. :class:`pyarrow.Schema`,
            :class:`arro3.core.Schema`).

        Returns
        -------
        CastingDataset

        """
        return CastingDataset(self._impl.cast(target_schema))

    def ingest_all(self) -> int:
        """Eagerly ingest all batches from the upstream source into the cache.

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
        >>> from batchcorder import CachedDataset
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = CachedDataset(table, 16 << 20, tmp, 64 << 20)
        >>> ds.ingest_all()
        1
        >>> ds.upstream_exhausted
        True

        """
        return self._impl.ingest_all()

    def close(self) -> None:
        """Close the dataset and destroy the underlying storage.

        This method clears the hybrid cache and destroys the disk storage,
        removing any unused files that were eagerly created.

        Returns
        -------
        None

        Examples
        --------
        >>> import tempfile, pyarrow as pa
        >>> from batchcorder import CachedDataset
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> tmp = tempfile.mkdtemp()
        >>> ds = CachedDataset(table, 16 << 20, tmp, 64 << 20)
        >>> ds.close()

        """
        return self._impl.close()


class CachedDatasetReader:
    """A single-use iterator handle for a :class:`CachedDataset`.

    Maintains an independent read position.  Multiple handles backed by the
    same dataset share the underlying cache; the upstream source is ingested
    lazily as needed.

    Once consumed via ``__arrow_c_stream__`` or by exhausting iteration the
    reader is marked closed and raises an error on further use.

    Notes
    -----
    Obtain a handle from :meth:`CachedDataset.reader` rather than constructing
    one directly.

    """

    def __init__(self, impl: _PyCachedDatasetReader):
        """Obtain via :meth:`CachedDataset.reader`."""
        self._impl = impl

    @property
    def schema(self) -> Any:
        """Arrow schema of batches produced by this reader.

        Returns
        -------
        arro3.core.Schema

        Raises
        ------
        IOError
            If the reader has already been consumed.

        """
        return self._impl.schema

    @property
    def closed(self) -> bool:
        """``True`` if this reader has been consumed.

        Returns
        -------
        bool

        """
        return self._impl.closed

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """Enable Arrow stream export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

        This dunder method should not be called directly, but enables zero-copy data
        transfer to other Python libraries that understand Arrow memory.

        Consumes the reader; subsequent calls raise an error.

        Parameters
        ----------
        requested_schema : object, optional
            Schema capsule to cast the stream to, or ``None``.

        Raises
        ------
        IOError
            If the reader has already been consumed.

        """
        return self._impl.__arrow_c_stream__(requested_schema)

    def __arrow_c_schema__(self) -> Any:
        """Enable Arrow schema export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

        This dunder method should not be called directly, but enables zero-copy data
        transfer to other Python libraries that understand Arrow memory.

        This allows Arrow consumers to inspect the data type of this
        :class:`CachedDatasetReader`.  Then the consumer can ask the producer (in
        ``__arrow_c_stream__``) to cast the exported data to a supported data type.

        Raises
        ------
        IOError
            If the reader has already been consumed.

        """
        return self._impl.__arrow_c_schema__()

    def __iter__(self) -> CachedDatasetReader:
        """Return self as the iterator."""
        return self

    def cast(self, target_schema: Any) -> Any:
        """Cast the reader to produce batches with the given schema.

        Mirrors :meth:`pyarrow.RecordBatchReader.cast`.  Returns a
        :class:`pyarrow.RecordBatchReader` that applies the cast as batches are
        read.  Consumes this reader.

        Parameters
        ----------
        target_schema : object
            Any Arrow schema-compatible object (e.g. :class:`pyarrow.Schema`,
            :class:`arro3.core.Schema`).

        Returns
        -------
        pyarrow.RecordBatchReader

        Raises
        ------
        IOError
            If the reader has already been consumed.

        """
        return self._impl.cast(target_schema)

    def __next__(self) -> Any:
        """Get the next batch from the reader."""
        return next(iter(self._impl))


class CastingDataset:
    """A replayable cast view of a :class:`CachedDataset`.

    Created by :meth:`CachedDataset.cast`.  Each call to ``__arrow_c_stream__``
    produces a fresh reader from the underlying cache with each batch cast to
    :attr:`schema`, so this object is **replayable** — DuckDB self-joins, ASOF
    joins, and other multi-scan consumers work correctly on it.

    Notes
    -----
    Obtain via :meth:`CachedDataset.cast` rather than constructing directly.

    """

    def __init__(self, impl: _PyCastingDataset):
        """Obtain via :meth:`CachedDataset.cast`."""
        self._impl = impl

    @property
    def schema(self) -> Any:
        """Arrow schema produced by this dataset after casting.

        Returns
        -------
        arro3.core.Schema

        """
        return self._impl.schema

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """Enable Arrow stream export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

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
        """Enable Arrow schema export via the `PyCapsule Interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_.

        Returns the target schema so consumers can inspect the post-cast type.

        """
        return self._impl.__arrow_c_schema__()

    def cast(self, target_schema: Any) -> CastingDataset:
        """Cast to a further target schema, returning a new :class:`CastingDataset`.

        Parameters
        ----------
        target_schema : object
            Any Arrow schema-compatible object.

        Returns
        -------
        CastingDataset

        """
        return CastingDataset(self._impl.cast(target_schema))
