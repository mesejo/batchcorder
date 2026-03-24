batchcorder
===========

A Rust-backed Python library for caching Arrow record-batch streams so they
can be replayed multiple times from a source that can only be read once.

Arrow ``RecordBatchReader`` is single-use: once consumed, it is gone.
**batchcorder** wraps any Arrow stream source and stores each batch in a
hybrid memory + disk cache (backed by `Foyer <https://github.com/foyer-rs/foyer>`_),
so multiple independent readers can replay the stream from any position.

----

.. toctree::
   :maxdepth: 2
   :caption: Tutorial — Learning

   tutorials/getting-started

.. toctree::
   :maxdepth: 2
   :caption: How-to Guides — Doing

   how-to/duckdb
   how-to/cache-config
   how-to/eviction

.. toctree::
   :maxdepth: 1
   :caption: Reference — Information

   api
   reference/api-overview
