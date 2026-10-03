#!/bin/bash
# Measure the local Rust edit/build iteration cycle on the current checkout.
#
# Usage:
#   bash scripts/bench-build.sh [--rounds N] [--clean]
#
#   --rounds N   repetitions per scenario (default 3)
#   --clean      also run `cargo clean` and time a cold `cargo build --lib`
#                (throws away the whole target/ cache; off by default)
#
# To compare two branches, run the same script on each. The script lives on
# the branch under test, so pull it out of git when measuring main:
#
#   git switch chore/faster-dev-builds && bash scripts/bench-build.sh
#   git show chore/faster-dev-builds:scripts/bench-build.sh > /tmp/bench-build.sh
#   git switch main && bash /tmp/bench-build.sh
#
# The warm-up step absorbs the one-off rebuild caused by switching branches
# with a different dev profile, so only steady-state numbers are reported.
#
# Scenarios:
#   alternate   maturin develop -> cargo build --lib -> cargo clippy (the
#               pre-commit hook command) -> maturin develop, with no source
#               change. Every step should be a no-op; on main each step
#               recompiles pyo3 and everything above it.
#   edit        append a one-line function to src/cached_dataset.rs, run
#               `maturin develop`, remove it, run `maturin develop` again.
#   sizes       size of the debug .so and of target/debug.
set -euo pipefail

rounds=3
clean=0
while [ $# -gt 0 ]; do
    case "$1" in
        --rounds) rounds="$2"; shift 2 ;;
        --clean) clean=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd "$(git rev-parse --show-toplevel)"
src=src/cached_dataset.rs
backup=$(mktemp)
cp "$src" "$backup"
trap 'cp "$backup" "$src"; rm -f "$backup"' EXIT

clippy_cmd=(cargo clippy --all-targets --all-features -- -Dclippy::all -D warnings -Aclippy::redundant_closure)

now_ms() { date +%s%N | cut -c1-13; }

# timed <label> <command...>: run quietly, print wall seconds.
timed() {
    local label="$1"; shift
    local start end
    start=$(now_ms)
    "$@" >/dev/null 2>&1
    end=$(now_ms)
    printf '  %-34s %6.2f s\n' "$label" "$(awk "BEGIN{print ($end-$start)/1000}")"
}

echo "branch: $(git branch --show-current)  commit: $(git rev-parse --short HEAD)"
echo "rustc: $(rustc --version)"
echo

echo "warm-up (not measured)"
uv run maturin develop --uv >/dev/null 2>&1
cargo build --lib >/dev/null 2>&1
"${clippy_cmd[@]}" >/dev/null 2>&1
uv run maturin develop --uv >/dev/null 2>&1
echo

echo "alternate: no source change, maturin <-> bare cargo"
for ((i = 1; i <= rounds; i++)); do
    echo " round $i"
    timed "maturin develop --uv" uv run maturin develop --uv
    timed "cargo build --lib" cargo build --lib
    timed "cargo clippy (pre-commit hook)" "${clippy_cmd[@]}"
    timed "maturin develop --uv" uv run maturin develop --uv
done
echo

echo "edit: one-line change in $src, then maturin develop"
for ((i = 1; i <= rounds; i++)); do
    echo " round $i"
    printf '\nfn __bench_probe_%d() -> u32 {\n    %d\n}\n' "$i" "$i" >>"$src"
    timed "add fn + maturin develop --uv" uv run maturin develop --uv
    cp "$backup" "$src"
    timed "remove fn + maturin develop --uv" uv run maturin develop --uv
done
echo

echo "sizes"
so=$(find python/batchcorder -maxdepth 1 -name '_batchcorder*.so' | head -n 1)
printf '  %-34s %6.1f MB\n' "installed .so" "$(awk "BEGIN{print $(stat -c %s "$so")/1048576}")"
printf '  %-34s %s\n' "target/debug" "$(du -sh target/debug | cut -f1)"
echo

if [ "$clean" -eq 1 ]; then
    echo "clean build (cargo clean first)"
    cargo clean
    timed "cargo build --lib (cold)" cargo build --lib
    uv run maturin develop --uv >/dev/null 2>&1
fi
