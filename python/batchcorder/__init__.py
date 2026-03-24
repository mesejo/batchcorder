"""Batchcorder: Hybrid memory+disk cached Arrow datasets."""

from importlib.metadata import version

from ._batchcorder import CachedDataset, CachedDatasetReader, CastingDataset


__all__ = [
    "CachedDataset",
    "CachedDatasetReader",
    "CastingDataset",
]


__version__: str = version("batchcorder")
