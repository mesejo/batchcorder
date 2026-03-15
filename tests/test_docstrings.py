"""Run the doctests embedded in batchcorder's compiled extension docstrings."""

import doctest

import batchcorder


def test_docstrings() -> None:
    results = doctest.testmod(
        batchcorder,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        verbose=False,
    )
    assert results.failed == 0, f"{results.failed} doctest(s) failed"
