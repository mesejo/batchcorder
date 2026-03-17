from importlib.metadata import version

from ._batchcorder import CachedDataset, CachedDatasetReader


__all__ = [
    "CachedDataset",
    "CachedDatasetReader",
]


__version__: str = version("batchcorder")
