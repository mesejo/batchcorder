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
