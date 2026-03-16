from importlib.metadata import version

from ._batchcorder import *  # noqa: F401, F403
from ._batchcorder import __all__  # noqa: F401


__version__ = version("batchcorder")
