# Metal GPU BFS Layout Optimization

## Overview

The Metal GPU Poseidon Merkle tree shader was modified to write digests directly in plonky2-fork's BFS (level-order) layout, eliminating a costly CPU-side layout conversion step. This optimization saves ~134MB of allocation and element-by-element copy per Merkle tree build at large degrees.

## Changes

### GPU writes directly in BFS layout

Previously the GPU shader used a linear layout with `subtree_digests_len = 2n - 1` (includes root slot), then a CPU function `convert_linear_to_plonky2_digests` remapped every element to plonky2-fork's BFS layout (`subtree_digests_len = 2(n - 1)`, no root slot).

The fix: pass `subtree_digests_len = 2(n - 1)` to the shader. The existing index formulas produce the correct BFS positions:
- Leaf: `(subtree_digests_len - n) + leaf_idx = (n - 2) + leaf_idx`
- Internal level l: `subtree_digests_len - 2n + (n >> level) = (n >> level) - 2`

The only shader code change was in the cap kernel: root's children moved from positions 1,2 to positions 0,1.

### Eliminated convert_linear_to_plonky2_digests

Removed the CPU-side conversion function that allocated a `Vec<HashOut>` (~134MB at d18) and copied element-by-element with index remapping. The GPU buffer is now read directly.

### Routing logic

```
tree_height < 13           -> CPU  (Metal dispatch overhead > compute benefit)
tree_height 13..=20        -> GPU linear+threadgroup
tree_height > 20           -> CPU  (GPU ~10-15% slower due to UMA bandwidth saturation)
cap_height == tree_height  -> CPU  (all-cap trees, no internal digests)
```

## Performance Results

Benchmark: `bench_degrees` test, `prove_min` across 3 iterations (d13-d17) or 2 iterations (d18+).

| Degree | CPU prove_min | Metal prove_min | Speedup |
|--------|--------------|-----------------|---------|
| d13 | 277ms | 217ms | **1.28x** |
| d14 | 563ms | 452ms | **1.25x** |
| d15 | 1.11s | 931ms | **1.20x** |
| d16 | 2.25s | 1.94s | **1.16x** |
| d17 | 4.93s | 4.24s | **1.16x** |
| d18 | 20.6s | 24.5s | 0.84x (CPU fallback) |
| d19 | 140s | 152s | 0.92x (CPU fallback) |

Metal is 16-28% faster for d13-d17. For d18+ the GPU is slower due to UMA memory bandwidth saturation — CPU Rayon parallelism across 16 independent subtrees wins.

### Why d18+ is slower on GPU

At d18 with tree_height=21 and cap_height=4:
- 16 independent subtrees, each with 131K leaves
- CPU: Rayon distributes 16 subtrees across all cores, each subtree also parallelized internally
- GPU: all work serialized through a single command buffer on one dispatch thread
- Digests buffer ~134MB + leaves ~134MB saturate the shared UMA memory bus

## How to Run

### Fast benchmark (d13-d17, ~45 seconds)

```bash
# CPU baseline
RUSTFLAGS="-Ctarget-cpu=native" cargo test --release \
  --test bench_degrees -- fast --nocapture --test-threads=1

# Metal GPU
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test bench_degrees -- fast --nocapture --test-threads=1
```

### Full benchmark (d13-d20, ~20+ minutes)

```bash
# CPU baseline
RUSTFLAGS="-Ctarget-cpu=native" cargo test --release \
  --test bench_degrees -- full --nocapture --test-threads=1

# Metal GPU
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test bench_degrees -- full --nocapture --test-threads=1
```

### Correctness tests

```bash
# Metal Merkle tree tests (tree_height 20-22)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test metal_coalesced -- --nocapture --test-threads=1

# Metal proof end-to-end test
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test metal_proof -- --nocapture --test-threads=1

# Metal Merkle correctness (unit test)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  -p plonky2 --lib -- test_merkle_trees_metal_poseidon_g64 --nocapture
```

### Recompile shaders (after editing .metal files)

```bash
cd plonky2/shaders

# Linear+threadgroup shader
xcrun -sdk macosx metal -c poseidon_merkle_hasher_linear_threadgroup.metal \
  -o poseidon_merkle_hasher_linear_threadgroup.air
xcrun -sdk macosx metallib poseidon_merkle_hasher_linear_threadgroup.air \
  -o poseidon_merkle_hasher_linear_threadgroup.metallib
rm poseidon_merkle_hasher_linear_threadgroup.air

# Coalesced shader
xcrun -sdk macosx metal -c poseidon_merkle_hasher_coalesced.metal \
  -o poseidon_merkle_hasher_coalesced.air
xcrun -sdk macosx metallib poseidon_merkle_hasher_coalesced.air \
  -o poseidon_merkle_hasher_coalesced.metallib
rm poseidon_merkle_hasher_coalesced.air
```

## Files Modified

| File | Change |
|------|--------|
| `plonky2/shaders/poseidon_merkle_hasher_linear_threadgroup.metal` | Cap kernel: children at 0,1 instead of 1,2 |
| `plonky2/shaders/poseidon_merkle_hasher_coalesced.metal` | Cap kernel: children at 0,1 instead of 1,2 |
| `plonky2/shaders/*.metallib` | Recompiled |
| `plonky2/src/hash/metal/merkle.rs` | BFS subtree_digests_len, removed convert_linear_to_plonky2_digests |
| `plonky2/src/hash/merkle_tree.rs` | Routing comment update |
| `plonky2/tests/bench_degrees.rs` | Split into fast (d13-d17) and full (d13-d20) groups |
