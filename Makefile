# Documentation build targets
#
#  make quarto   — render .qmd files in docs/quarto/ to Markdown in docs/source/
#  make sphinx   — build the Sphinx HTML site to docs/build/html/
#  make docs     — run both steps in sequence

QUARTO_SRC  := docs/quarto
SPHINX_SRC  := docs/source
SPHINX_OUT  := docs/build/html

.PHONY: quarto sphinx docs clean-docs

quarto:
	quarto render $(QUARTO_SRC)/

sphinx:
	uv run sphinx-build -b html $(SPHINX_SRC) $(SPHINX_OUT)

docs: quarto sphinx

clean-docs:
	rm -rf $(SPHINX_OUT) $(SPHINX_SRC)/_autoapi
	find $(SPHINX_SRC) -name '*.md' -delete

# DuckDB loadable extension
#
#  make duckdb-ext       — build target/duckdb-extension/batchcorder.duckdb_extension
#  make test-duckdb-ext  — LOAD it into DuckDB 1.3.2, 1.5.0 and 2.0.1 and run the e2e tests
#
# The crate is built on its own (-p): its loadable-extension feature must not
# share a feature resolve with batchcorder-duckdb's bundled test DuckDB.

# The footer script needs only the standard library.
PYTHON              ?= python3
# Each DuckDB runs in a throwaway environment, apart from the project's venv.
DUCKDB_PY            = uv run --no-project --with duckdb$(if $(1),==$(1)) --with pyarrow --with pytest python
DUCKDB_EXT_DIR      := target/duckdb-extension
DUCKDB_EXT          := $(DUCKDB_EXT_DIR)/batchcorder.duckdb_extension
# Minimum DuckDB C API version, read from the entrypoint's min_duckdb_version
# so the footer and the API the extension requests cannot disagree.
DUCKDB_EXT_CAPI     := $(shell sed -n 's/.*min_duckdb_version = "\(v[^"]*\)".*/\1/p' crates/batchcorder-duckdb-ext/src/lib.rs)
DUCKDB_EXT_VERSION  := $(shell sed -n 's/^version = "\(.*\)"/\1/p' crates/batchcorder-duckdb-ext/Cargo.toml)
DUCKDB_EXT_LIB      := target/release/libbatchcorder_duckdb_ext$(if $(filter Darwin,$(shell uname -s)),.dylib,.so)
# DuckDB's name for this machine, e.g. linux_amd64; override to cross-label.
DUCKDB_PLATFORM     ?= $(shell $(call DUCKDB_PY) -c "import duckdb; print(duckdb.execute('PRAGMA platform').fetchone()[0])" 2>/dev/null)
# 1.3.2 is the oldest that accepts the default Linux label; 1.2.x reports
# linux_amd64_gcc4 (see the crate docs).
DUCKDB_TEST_VERSIONS := 1.3.2 1.5.0 2.0.1

.PHONY: duckdb-ext test-duckdb-ext

duckdb-ext:
	cargo build --release -p batchcorder-duckdb-ext
	platform='$(DUCKDB_PLATFORM)'; \
	if [ -z "$$platform" ]; then \
		echo "could not detect the DuckDB platform; set DUCKDB_PLATFORM, e.g. linux_amd64" >&2; \
		exit 1; \
	fi; \
	$(PYTHON) scripts/duckdb_ext/append_metadata.py $(DUCKDB_EXT_LIB) $(DUCKDB_EXT) \
		--name batchcorder --platform "$$platform" \
		--duckdb-version $(DUCKDB_EXT_CAPI) --extension-version $(DUCKDB_EXT_VERSION)

test-duckdb-ext: duckdb-ext
	for v in $(DUCKDB_TEST_VERSIONS); do \
		$(call DUCKDB_PY,$$v) -m pytest crates/batchcorder-duckdb-ext/tests \
			--noconftest -p no:cacheprovider || exit 1; \
	done
