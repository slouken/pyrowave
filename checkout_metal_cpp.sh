#!/bin/bash

# Fetches Apple's metal-cpp headers, needed by the Metal decoder backend.
# metal-cpp is header-only, so this just unpacks it next to the sources.
#
# The pinned release targets an older SDK than the one you are likely building
# with; that is fine, as the decoder only uses long-stable Metal API. Bump the
# version here if a newer bundle is needed.

set -e

METAL_CPP_VERSION=metal-cpp_macOS15.2_iOS18.2
METAL_CPP_URL=https://developer.apple.com/metal/cpp/files/$METAL_CPP_VERSION.zip

if [ -d metal-cpp ]; then
	echo "metal-cpp already present, nothing to do."
	exit 0
fi

TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

echo "Downloading $METAL_CPP_VERSION ..."
curl -L --fail -o "$TMPDIR_LOCAL/metal-cpp.zip" "$METAL_CPP_URL"

unzip -q "$TMPDIR_LOCAL/metal-cpp.zip" -d "$TMPDIR_LOCAL"

# The archive contains a single top level metal-cpp directory.
mv "$TMPDIR_LOCAL/metal-cpp" metal-cpp

echo "metal-cpp checked out."
