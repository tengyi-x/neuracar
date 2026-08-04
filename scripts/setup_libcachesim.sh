#!/usr/bin/env bash
# Clones and builds libCacheSim (requires cmake, a C/C++ toolchain, and libzstd/liblz4 dev headers).
# See https://github.com/cacheMon/libCacheSim for platform-specific prerequisites.
set -euo pipefail

REPO_DIR="${1:-$(dirname "$0")/../third_party/libCacheSim}"

git clone --depth 1 https://github.com/cacheMon/libCacheSim.git "$REPO_DIR"
cd "$REPO_DIR"
mkdir -p _build && cd _build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "Built libCacheSim in $REPO_DIR/_build. Baselines (LRU, LFU, ARC, LeCaR, LHD, LRB) run via the 'cachesim' CLI."
