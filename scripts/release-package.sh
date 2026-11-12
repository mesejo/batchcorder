#!/bin/bash
# Print `package=<name>` for the release workflows ($GITHUB_OUTPUT format).
#
# Usage: release-package.sh <event> <tag> <input-package>
#
# Maps a release tag to a package and checks that the tag's version matches
# the crate version:
#   batchcorder-vX.Y.Z             -> batchcorder
#   batchcorder-datafusion-vX.Y.Z  -> batchcorder-datafusion
# <tag> is the published release's tag, or for a workflow_dispatch the tag the
# run was dispatched on (empty when dispatched on a branch).  A dispatch must
# run on a release tag, and <input-package> must be the package the tag names,
# so a dispatch publishes exactly what a release of that tag would.
set -euo pipefail

event="$1"
tag="$2"
input_package="$3"

# crate_of <package>: the Cargo package that builds the wheel.
crate_of() {
    case "$1" in
        batchcorder) echo batchcorder ;;
        batchcorder-datafusion) echo batchcorder-datafusion-py ;;
        *) echo "Unknown package: '$1'" >&2; exit 1 ;;
    esac
}

if [ "$event" != "release" ] && [ -z "$tag" ]; then
    echo "Dispatch ci-release.yaml on a release tag, not a branch" >&2
    exit 1
fi

# The longer prefix first: batchcorder-v* does not match
# batchcorder-datafusion-v*, but keep the order explicit.
case "$tag" in
    batchcorder-datafusion-v*) package=batchcorder-datafusion ;;
    batchcorder-v*) package=batchcorder ;;
    *)
        echo "Release tag '$tag' names no package; expected" \
            "batchcorder-vX.Y.Z or batchcorder-datafusion-vX.Y.Z" >&2
        exit 1
        ;;
esac

if [ "$event" != "release" ] && [ "$input_package" != "$package" ]; then
    echo "Tag '$tag' releases $package, but the dispatch asked for" \
        "'$input_package'" >&2
    exit 1
fi

tag_version="${tag#"$package"-v}"
crate=$(crate_of "$package")
crate_version=$(
    cargo metadata --no-deps --format-version 1 |
        python3 -c 'import json, sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == sys.argv[1]))' "$crate"
)
if [ "$tag_version" != "$crate_version" ]; then
    echo "Release tag '$tag' says $tag_version, but crate $crate is $crate_version" >&2
    exit 1
fi

echo "package=$package"
