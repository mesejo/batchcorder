"""Check that a DataFusion host is built on the same DataFusion major as this wheel."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


# The provider crosses into the host through datafusion-ffi, whose structs are
# only compatible within one DataFusion major.  Each entry maps a host
# distribution to the half-open range [low, high) of its versions built on the
# same DataFusion major as this wheel (datafusion-ffi in Cargo.toml).
#
# Keep this table and the compatibility table in README.md in sync.
SUPPORTED_HOSTS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "datafusion": ((55,), (56,)),
    "xorq-datafusion": ((0, 3), (0, 4)),
}

# Top-level Python package of each host's session class -> its distribution.
HOST_PACKAGES: dict[str, str] = {
    "datafusion": "datafusion",
    "xorq_datafusion": "xorq-datafusion",
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


def _format(release: tuple[int, ...]) -> str:
    return ".".join(map(str, release))


def check_host(
    session: object,
    get_version: Callable[[str], str] = version,
) -> None:
    """
    Raise if ``session`` belongs to a host built on another DataFusion major.

    Called by ``TeeTableProvider.__datafusion_table_provider__`` before any
    FFI struct of the host is touched.  The host is identified by the package
    of the session's class.  A codec capsule, or a session from an unknown
    host, cannot be identified here; the provider then checks the
    datafusion-ffi version of the host's codec instead.

    Parameters
    ----------
    session
        The ``session`` argument the host passed to
        ``__datafusion_table_provider__``.
    get_version
        Callable mapping a distribution name to its installed version and
        raising :class:`importlib.metadata.PackageNotFoundError` when it is
        not installed.  Defaults to :func:`importlib.metadata.version`.

    Raises
    ------
    RuntimeError
        If the host's installed version is outside its supported range.

    """
    package = type(session).__module__.partition(".")[0]
    dist = HOST_PACKAGES.get(package)
    if dist is None:
        return
    try:
        installed = get_version(dist)
    except PackageNotFoundError:
        return
    low, high = SUPPORTED_HOSTS[dist]
    if low <= _release(installed) < high:
        return
    raise RuntimeError(
        f"TeeTableProvider cannot be registered with {dist} {installed}: "
        f"batchcorder-datafusion is built on DataFusion {DATAFUSION_MAJOR} and "
        f"needs {dist}>={_format(low)},<{_format(high)}.  Install the "
        f"batchcorder-datafusion release built on the same DataFusion major "
        f"as {dist} {installed}, or a {dist} in that range."
    )
