"""Import-time check that the installed DataFusion hosts match this wheel."""

from __future__ import annotations

import re
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


# The provider crosses into the host through datafusion-ffi, whose structs are
# only compatible within one DataFusion major.  Each entry maps a host
# distribution to the half-open range [low, high) of its versions built on the
# same DataFusion major as this wheel (datafusion-ffi in Cargo.toml).
#
# Keep this table and the extras in pyproject.toml in sync.
SUPPORTED_HOSTS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "datafusion": ((55,), (56,)),
    "xorq-datafusion": ((0, 3), (0, 4)),
}

DATAFUSION_MAJOR = 55


def _release(text: str) -> tuple[int, ...]:
    """
    Return the leading numeric release segment of a version string.

    Parameters
    ----------
    text
        A version such as ``"55.0.0"`` or ``"0.3.0rc1"``.

    Returns
    -------
    tuple[int, ...]
        The release numbers, e.g. ``(55, 0, 0)`` or ``(0, 3, 0)``.

    """
    match = re.match(r"\d+(?:\.\d+)*", text)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group().split("."))


def mismatched_hosts(
    get_version: Callable[[str], str] = version,
) -> list[tuple[str, str, str]]:
    """
    List the installed hosts whose version is outside the supported range.

    Parameters
    ----------
    get_version
        Callable mapping a distribution name to its installed version and
        raising :class:`importlib.metadata.PackageNotFoundError` when it is
        not installed.  Defaults to :func:`importlib.metadata.version`.

    Returns
    -------
    list[tuple[str, str, str]]
        ``(distribution, installed version, supported range)`` for each
        installed host outside its range.

    """
    found = []
    for dist, (low, high) in SUPPORTED_HOSTS.items():
        try:
            installed = get_version(dist)
        except PackageNotFoundError:
            continue
        release = _release(installed)
        if not low <= release < high:
            supported = f">={'.'.join(map(str, low))},<{'.'.join(map(str, high))}"
            found.append((dist, installed, supported))
    return found


def warn_on_mismatched_hosts() -> None:
    """Warn about each installed host built on another DataFusion major."""
    for dist, installed, supported in mismatched_hosts():
        warnings.warn(
            f"batchcorder-datafusion is built against DataFusion "
            f"{DATAFUSION_MAJOR} and needs {dist}{supported}, but {dist} "
            f"{installed} is installed.  Registering a TeeTableProvider with "
            f"it can fail or crash; install a batchcorder-datafusion release "
            f"built for that host's DataFusion major.",
            RuntimeWarning,
            stacklevel=2,
        )
