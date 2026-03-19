#!/bin/bash

# Script to build documentation with error handling and security checks
# Usage: ./scripts/build-docs.sh

set -euo pipefail

# Function to handle errors
handle_error() {
    echo "Error: $1" >&2
    exit 1
}

# Function to verify checksum
verify_checksum() {
    local file=$1
    local expected_checksum=$2
    local actual_checksum

    actual_checksum=$(sha256sum "$file" | awk '{print $1}')

    if [ "$actual_checksum" != "$expected_checksum" ]; then
        handle_error "Checksum verification failed for $file"
    fi
}

# Install Python documentation dependencies
echo "Installing Python documentation dependencies..."
pip install -r docs/requirements.txt || handle_error "Failed to install Python dependencies"

# Download and verify Quarto
echo "Downloading Quarto..."
curl -fsSL -o quarto.tar.gz "https://github.com/quarto-dev/quarto-cli/releases/download/v1.8.24/quarto-1.8.24-linux-amd64.tar.gz" || handle_error "Failed to download Quarto"

# Verify checksum (replace with actual checksum)
echo "Verifying Quarto checksum..."
verify_checksum "quarto.tar.gz" "expected_sha256_checksum_here" || handle_error "Quarto checksum verification failed"

# Extract Quarto
echo "Extracting Quarto..."
tar -xzf quarto.tar.gz || handle_error "Failed to extract Quarto"

# Render .qmd → Markdown in docs/source/
echo "Rendering Quarto files..."
./quarto-1.8.24/bin/quarto render docs/quarto/ || handle_error "Failed to render Quarto files"

# Build Sphinx HTML into the RTD output directory
echo "Building Sphinx documentation..."
sphinx-build -b html docs/source/ "$READTHEDOCS_OUTPUT/html" || handle_error "Failed to build Sphinx documentation"

echo "Documentation build completed successfully!"
