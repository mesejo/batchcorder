#!/bin/bash
set -euo pipefail

# Check if bump type is provided
if [ -z "${1:-}" ]; then
    echo "Error: Bump type not provided."
    echo "Usage: $0 <major|minor|patch>"
    exit 1
fi

bump_type="$1"

if [[ "$bump_type" != "major" && "$bump_type" != "minor" && "$bump_type" != "patch" ]]; then
    echo "Error: Bump type must be one of: major, minor, patch"
    exit 1
fi

# Step 1: Ensure you're on upstream main
echo "Step 1: Switching to upstream main..."
git switch main && git pull

# Step 2: Bump the version in Cargo.toml
echo "Step 2: Bumping $bump_type version..."
cargo set-version --bump "$bump_type"
version_number=$(grep '^version' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
echo "New version: $version_number"

# Step 3: Create a release branch
echo "Step 3: Creating release branch..."
git switch --create="release-$version_number"

# Step 4: Create commit
echo "Step 4: Creating commit..."
git add --update && git commit -m "release: $version_number"

# Step 5: Push the new branch
echo "Step 5: Pushing the new branch..."
git push --set-upstream origin "release-$version_number"

echo "Release process completed successfully!"
