#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

compile_one() {
    local src="$1"
    local out="${src%.metal}.metallib"
    local air="${src%.metal}.air"

    echo "Compiling $src -> $out"
    xcrun -sdk macosx metal -Ofast -ffast-math -fmetal-math-mode=fast -ffp-contract=fast -I "$SCRIPT_DIR" -c "$src" -o "$air"
    xcrun -sdk macosx metallib "$air" -o "$out"
    rm -f "$air"
}

compile_one poseidon_merkle_hasher.metal
compile_one poseidon_merkle_hasher_threadgroup.metal
compile_one poseidon_merkle_hasher_linear.metal
compile_one poseidon_merkle_hasher_linear_threadgroup.metal
compile_one poseidon_merkle_hasher_coalesced.metal

echo "Done."
