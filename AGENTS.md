# batchcorder Repository

This repository contains batchcorder, a Rust-backed Python library for caching Arrow record-batch streams so they can be replayed multiple times from a source that can only be read once.

## Architecture

The project consists of:

- **Python API** (`python/`): Python interface and bindings
- **Rust Core** (`src/`): Core implementation in Rust using PyO3 for Python bindings
- **Tests** (`tests/`): Test suite for verifying functionality
- **Documentation** (`docs/`): Project documentation

The Rust core implements the `StreamCache` functionality using:
- Arrow C Stream interface for compatibility
- Foyer for hybrid memory/disk caching
- PyO3 for Python interoperability

## Running Tests

Run all tests using pytest:

```bash
uv run pytest
```

Run specific test files or modules:

```bash
uv run pytest tests/test_module.py
```

## Running Debug Builds

Use debug builds when developing:

```bash
# Install dependencies without building the extension
uv sync --no-install-project --dev

# For docs development, include docs group
uv sync --no-install-project --dev --group docs

# Build the extension
uv run maturin develop --uv
```

## Development Guidelines

- All changes must be tested. If you're not testing your changes, you're not done.
- Look to see if your tests could go in an existing file before adding a new file for your tests.
- Get your tests to pass. If you didn't run the tests, your code does not work.
- Follow existing code style. Check neighboring files for patterns.
- Always run `uv run pre-commit run --all-files` at the end of a task, after every rebase, after addressing any review comment, and before pushing any code.
- Avoid writing significant amounts of new code. This is often a sign that we're missing an existing method or mechanism that could help solve the problem. Look for existing utilities first.
- Try hard to avoid patterns that require `panic!`, `unreachable!`, or `.unwrap()`. Instead, try to encode those constraints in the type system. Don't be afraid to write code that's more verbose or requires largeish refactors if it enables you to avoid these unsafe calls.
- Prefer let chains (`if let` combined with `&&`) over nested `if let` statements to reduce indentation and improve readability. At the end of a task, always check your work to see if you missed opportunities to use `let` chains.
- If you *have* to suppress a Clippy lint, prefer to use `#[expect()]` over `[allow()]`, where possible. But if a lint is complaining about unused/dead code, it's usually best to just delete the unused code.
- Use comments purposefully. Don't use comments to narrate code, but do use them to explain invariants and why something unusual was done a particular way.
