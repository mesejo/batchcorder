"""Sphinx configuration for batchcorder documentation."""

import logging
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

project = "batchcorder"
author = "batchcorder contributors"
release = "0.1"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

extensions = [
    "myst_parser",  # Parse .md files written/generated for Sphinx
    "sphinx.ext.napoleon",  # NumPy / Google docstring support
    "sphinx.ext.autodoc",  # Included for completeness; API gen uses autoapi
    "sphinx.ext.intersphinx",  # Cross-ref to PyArrow / Python standard library
    "autoapi.extension",  # Static-analysis API docs (works without .so)
    "sphinx_autodoc_typehints",
    "sphinx_immaterial",
]

# ---------------------------------------------------------------------------
# AutoAPI — static analysis of .pyi stubs (no compiled extension needed)
# ---------------------------------------------------------------------------
# We target only *.pyi files so autoapi reads the generated type stubs
# (python/batchcorder/__init__.pyi) instead of trying to import the Rust
# extension module (_batchcorder.abi3.so), which is unavailable on RTD.

autoapi_dirs = [str(Path(__file__).parents[2] / "python")]
autoapi_type = "python"
autoapi_file_patterns = ["*.pyi"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autoapi_add_toctree_entry = False  # We include it explicitly in index.rst
autoapi_python_class_content = "both"
autoapi_member_order = "source"
autoapi_keep_files = False

# ---------------------------------------------------------------------------
# Napoleon (NumPy docstring style)
# ---------------------------------------------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False

# ---------------------------------------------------------------------------
# MyST-Parser
# ---------------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}

# ---------------------------------------------------------------------------
# HTML theme — sphinx-immaterial
# ---------------------------------------------------------------------------


class _SuppressFilter(logging.Filter):
    def __init__(self, pattern):
        super().__init__()
        self._re = re.compile(pattern)

    def filter(self, record):
        return not self._re.search(record.getMessage())


logging.getLogger("sphinx.sphinx_immaterial.apidoc.python.parameter_objects").addFilter(
    _SuppressFilter(
        r"Parameter name '(reader|memory_capacity|disk_path|disk_capacity)' does not match any of the parameters"
    )
)

html_theme = "sphinx_immaterial"
html_title = "batchcorder"

html_theme_options = {
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "indigo",
            "accent": "indigo",
            "toggle": {
                "icon": "material/brightness-7",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "indigo",
            "accent": "indigo",
            "toggle": {
                "icon": "material/brightness-4",
                "name": "Switch to light mode",
            },
        },
    ],
    "site_url": "https://batchcorder.readthedocs.io/",
    "repo_url": "https://github.com/yourusername/multirecord",
    "repo_name": "batchcorder",
    "icon": {
        "repo": "fontawesome/brands/github",
        "logo": "material/archive-arrow-down",
    },
    "features": [
        "navigation.expand",
        "navigation.top",
        "search.highlight",
        "toc.follow",
    ],
}
