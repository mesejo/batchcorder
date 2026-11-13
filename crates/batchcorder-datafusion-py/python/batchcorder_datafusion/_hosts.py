"""Check that a DataFusion host is built on the same DataFusion major as this wheel."""

from __future__ import annotations

import re
import sys
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

# Top-level Python package of each host -> its distribution.  A session class
# belongs to a host when its module is in the package, or when the package
# exports it (see `_host_distribution`).
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


def _exports(package: str, cls: type) -> bool:
    """
    Tell whether ``package`` or one of its imported submodules exports ``cls``.

    Parameters
    ----------
    package
        A host's top-level Python package.
    cls
        The class of the session the host passed.

    Returns
    -------
    bool
        ``True`` if an imported module of ``package`` has ``cls`` as an
        attribute.

    """
    prefix = package + "."
    # Only modules already imported: the host imported its own package to
    # build the session, and importing one here could have side effects.
    for name, module in list(sys.modules.items()):
        if module is None or not (name == package or name.startswith(prefix)):
            continue
        if any(value is cls for value in vars(module).values()):
            return True
    return False


def _host_distribution(session: object) -> str | None:
    """
    Return the distribution of the host that ``session`` belongs to.

    Parameters
    ----------
    session
        The ``session`` argument the host passed to
        ``__datafusion_table_provider__``.

    Returns
    -------
    str | None
        The host's distribution, or ``None`` for an unknown host.

    """
    cls = type(session)
    dist = HOST_PACKAGES.get(cls.__module__.partition(".")[0])
    if dist is not None or cls.__module__ in sys.modules:
        return dist
    # A native (PyO3) class reports the module name given at build time,
    # which need not be a real module: xorq-datafusion's SessionContext says
    # `let`.  Find the host package that exports the class itself.  A class
    # from a real module (a capsule is `builtins.PyCapsule`, which hosts
    # re-export) is never looked up this way.
    for package, dist in HOST_PACKAGES.items():
        if _exports(package, cls):
            return dist
    return None


def check_host(
    session: object,
    get_version: Callable[[str], str] = version,
) -> None:
    """
    Raise if ``session`` belongs to a host built on another DataFusion major.

    Called by ``TeeTableProvider.__datafusion_table_provider__`` before any
    FFI struct of the host is touched.  The host is identified by the package
    of the session's class, or by the host package that exports that class.
    A codec capsule, or a session from an unknown host, cannot be identified
    here; the provider then checks the datafusion-ffi version of the host's
    codec instead.

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
    dist = _host_distribution(session)
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
