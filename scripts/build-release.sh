#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <batchcorder|batchcorder-datafusion> <major|minor|patch|X.Y.Z>"
    echo
    echo "batchcorder-datafusion's major version is the DataFusion major it is"
    echo "built on; move to a new DataFusion major with an explicit X.Y.Z."
}

if [ $# -ne 2 ]; then
    usage
    exit 1
fi

package="$1"
bump="$2"

# The Cargo package whose version is the wheel's version.
case "$package" in
    batchcorder) crate=batchcorder ;;
    batchcorder-datafusion) crate=batchcorder-datafusion-py ;;
    *)
        echo "Error: unknown package '$package'"
        usage
        exit 1
        ;;
esac

case "$bump" in
    major | minor | patch) set_version_args=(--bump "$bump") ;;
    [0-9]*.[0-9]*.[0-9]*) set_version_args=("$bump") ;;
    *)
        echo "Error: '$bump' is neither a bump type nor a version"
        usage
        exit 1
        ;;
esac

# Step 1: Ensure you're on upstream main
echo "Step 1: Switching to upstream main..."
git switch main && git pull

# Step 2: Bump the version of the package's crate.  The root Cargo.toml is a
# virtual workspace, so name the package explicitly.
echo "Step 2: Setting $package version ($bump)..."
cargo set-version --package "$crate" "${set_version_args[@]}"
# pkgid prints e.g. path+file:///.../crates/batchcorder-py#batchcorder@0.1.4
version_number=$(cargo pkgid --package "$crate" | sed 's/.*[@#]//')
echo "New version: $version_number"

# Step 3: Create a release branch
echo "Step 3: Creating release branch..."
git switch --create="release-$package-$version_number"

# Step 4: Create commit
echo "Step 4: Creating commit..."
git add --update && git commit -m "release($package): $version_number"

# Step 5: Push the new branch
echo "Step 5: Pushing the new branch..."
git push --set-upstream origin "release-$package-$version_number"

echo "Release branch pushed.  After it merges, publish the GitHub release"
echo "with tag $package-v$version_number."
