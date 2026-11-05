#!/bin/bash
# Print `package=<name>` for the release workflows ($GITHUB_OUTPUT format).
#
# Usage: release-package.sh <event> <tag> <input-package>
#
# A published release maps its tag to a package and checks that the tag's
# version matches the crate version:
#   batchcorder-vX.Y.Z             -> batchcorder
#   batchcorder-datafusion-vX.Y.Z  -> batchcorder-datafusion
# Any other event (workflow_dispatch) uses <input-package> as given.
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

if [ "$event" != "release" ]; then
    crate_of "$input_package" > /dev/null
    echo "package=$input_package"
    exit 0
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
