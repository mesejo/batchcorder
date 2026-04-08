# batchcorder Repository

This repository contains batchcorder, a Rust-backed Python library for caching Arrow record-batch streams so they can be replayed multiple times from a source that can only be read once.

## 📦 Architecture

The project consists of:

- **Python API** (`python/`): Python interface and bindings
- **Rust Core** (`src/`): Core implementation in Rust using PyO3 for Python bindings
- **Tests** (`tests/`): Test suite for verifying functionality
- **Documentation** (`docs/`): Project documentation

The Rust core implements the `StreamCache` functionality using:
- Arrow C Stream interface for compatibility
- Foyer for hybrid memory/disk caching
- PyO3 for Python interoperability

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Rust toolchain (stable)
- `uv` for dependency management

### Setup Commands

```bash
# Install dependencies without building the extension
uv sync --no-install-project --dev

# For docs development, include docs group
uv sync --no-install-project --dev --group docs

# Build the extension
uv run maturin develop --uv
```

### Environment variables

Set `UV_NO_SYNC=1` in your shell so that `uv run` never re-syncs the
virtual environment on every invocation:

```bash
export UV_NO_SYNC=1
```

Add this to your shell profile (`.bashrc`, `.zshrc`, etc.) to make it
permanent for this project.

## 🧪 Testing

### Run Tests

```bash
# Run all tests
uv run pytest

# Run specific test files or modules
uv run pytest tests/test_module.py
```

### Testing Guidelines

- All changes must be tested
- Look to see if your tests could go in an existing file before adding a new file
- Get your tests to pass before committing
- Run `uv run pre-commit run --all-files` before pushing code

## 📝 Code Style

### Rust Guidelines

- Avoid patterns that require `panic!`, `unreachable!`, or `.unwrap()`
- Prefer let chains (`if let` combined with `&&`) over nested `if let` statements
- Use `#[expect()]` over `[allow()]` for suppressing Clippy lints
- Use comments purposefully to explain invariants and unusual decisions

### Python Guidelines

- Follow existing code style in neighboring files
- Use type hints where appropriate
- Keep functions small and focused

### Code Examples

```rust
// ✅ Good: Proper error handling with let chains
if let Some(value) = optional_value && value.is_valid() {
    // Handle valid value
    process_value(value);
}

// ❌ Bad: Nested if let statements
if let Some(value) = optional_value {
    if value.is_valid() {
        // Handle valid value
        process_value(value);
    }
}
```

```python
# ✅ Good: Type hints and clear function naming
def calculate_total(items: List[Item]) -> float:
    """Calculate the total price of items."""
    return sum(item.price for item in items)

# ❌ Bad: No type hints, vague function name
def process(x):
    # Process something
    pass
```

## 📂 Project Structure

```
batchcorder/
├── python/          # Python API and bindings
├── src/             # Rust core implementation
├── tests/           # Test suite
├── docs/            # Documentation
├── CONTRIBUTING.md  # Contribution guidelines (this file)
└── README.md         # Human-readable overview
```

## 🔧 Development Workflow

1. **Create a feature branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Follow code style guidelines
3. **Run tests**: `uv run pytest`
4. **Run pre-commit**: `uv run pre-commit run --all-files`
5. **Commit changes**: Use clear, descriptive commit messages
6. **Push and create PR**: Get code review before merging

## 🚢 Releasing

### Prerequisites

Install `cargo-edit` for version bumping:

```bash
cargo install cargo-edit
```

### Release Process

Use the release script, passing the bump type as an argument:

```bash
bash scripts/build-release.sh <major|minor|patch>
```

This will:
1. Switch to upstream `main` and pull the latest changes
2. Bump the version in `Cargo.toml` using `cargo set-version --bump`
3. Create a `release-<version>` branch
4. Commit the version change
5. Push the branch to upstream

## 🛑 Boundaries

### Never Touch

- `.env*` files - Environment variables
- `target/` - Rust build artifacts
- `__pycache__/` - Python cache files
- `node_modules/` - Node.js dependencies (if any)
- `*.lock` files - Dependency lock files

### Always Do

- Write tests for new functionality
- Run existing tests before committing
- Follow code style guidelines
- Document public APIs

### Ask First

- Major architectural changes
- Breaking changes to public APIs
- Changes to CI/CD configuration
- Dependency updates

## 📖 Documentation

### Dependencies

Documentation requires:
- Quarto for rendering .qmd files to Markdown
- Sphinx for building HTML documentation
- Python dependencies listed in `docs/requirements.txt`

Install documentation dependencies:
```bash
uv sync --no-install-project --dev --group docs
```

### Building Documentation

```bash
# Render Quarto files to Markdown
make quarto

# Build Sphinx HTML documentation
make sphinx

# Build both (quarto + sphinx)
make docs

# Clean documentation build artifacts
make clean-docs
```

### Quarto Usage

- Quarto source files are in `docs/quarto/`
- Configuration in `docs/quarto/_quarto.yml`
- Code execution is disabled in Quarto (requires compiled Rust extension)
- Output directory: `docs/source/`

### Documentation Structure

```
docs/
├── quarto/           # Quarto source files (.qmd)
│   ├── how-to/       # How-to guides
│   ├── reference/    # Reference documentation
│   ├── tutorials/    # Tutorials
│   └── _quarto.yml   # Quarto configuration
├── source/           # Generated Markdown files
├── build/            # Built HTML documentation
├── requirements.txt  # Documentation dependencies
└── Makefile           # Documentation build targets
```

### Documentation Guidelines

- Write for a developer audience
- Focus on clarity and practical examples
- Keep documentation up-to-date with code changes
- Use Markdown format for documentation files
- Follow the structure defined in Quarto configuration

## 🤖 Agent-Specific Instructions

### For Coding Agents

- You are a Rust/Python developer working on the batchcorder project
- Focus on the Rust core (`src/`) and Python bindings (`python/`)
- Follow the existing architecture and patterns
- Never modify files outside the project scope

### For Documentation Agents

- You are a technical writer for the batchcorder project
- Read code from `src/` and `python/` to understand functionality
- Write documentation in `docs/quarto/` using Quarto format
- Follow the documentation guidelines above

### For Testing Agents

- You are a QA engineer for the batchcorder project
- Write tests in the `tests/` directory
- Ensure tests cover edge cases and error conditions
- Never remove failing tests without authorization

## 🔄 Version Control

### Git Workflow

- Use feature branches for new functionality
- Use fix branches for bug fixes
- Create draft PRs for work-in-progress
- Require at least one approval before merging
- Squash merge for clean history

### Commit Message Format

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

Where:
- `type`: feat, fix, docs, style, refactor, perf, test, chore
- `scope`: module or component affected
- `subject`: brief description of changes
- `body`: detailed explanation (optional)
- `footer`: breaking changes or issue references (optional)

## 🔒 Security

### Security Guidelines

- Never commit secrets or API keys
- Keep dependencies up-to-date
- Follow Rust's security best practices
- Use secure coding practices in Python

## 📚 Resources

- [Arrow Documentation](https://arrow.apache.org/docs/)
- [PyO3 Documentation](https://pyo3.rs/)
- [Foyer Documentation](https://foyer.readthedocs.io/)
- [Rust Documentation](https://doc.rust-lang.org/)
- [Python Documentation](https://docs.python.org/3/)
- [Quarto Documentation](https://quarto.org/docs/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)

## 🤝 Contributing

We welcome contributions! Please:

1. Read the CONTRIBUTING.md file thoroughly
2. Follow the development guidelines
3. Write tests for your changes
4. Submit a pull request with clear description
5. Be responsive to feedback and reviews

## 📜 License

This project is licensed under the Apache License. See the LICENSE file for details.
