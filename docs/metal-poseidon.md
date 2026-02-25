# Metal Poseidon GPU Acceleration

Metal-accelerated Merkle tree building for the Poseidon hash function on Apple Silicon,
integrated into `plonky2-fork` behind the `metal` feature flag.

---

## What Was Changed

### New feature flag

**`plonky2/Cargo.toml`** — added `metal` feature and `metal` crate dependency:

```toml
[features]
metal = ["dep:metal"]

[dependencies]
metal = { version = "0.27", optional = true }
```

### New shader files

**`plonky2/src/hash/metal/shaders/`** — MSL Metal shaders for Poseidon hashing:

| File | Purpose |
|------|---------|
| `poseidon_hash_leaves_linear_threadgroup.metal` | Hash raw field element leaves → digest layer |
| `poseidon_hash_tree_level_linear_threadgroup.metal` | Hash one internal tree level |
| `poseidon_hash_caps_linear_threadgroup.metal` | Hash cap nodes (roots of each subtree) |

The shaders use a **linear memory layout**: all nodes at a given level are stored
contiguously. This maximises GPU memory coalescing.

### New Rust modules

**`plonky2/src/hash/metal/`** — all gated behind `#[cfg(feature = "metal")]`:

| File | Purpose |
|------|---------|
| `mod.rs` | Module root; re-exports `RUNTIME` |
| `runtime.rs` | Metal device/queue init, buffer allocation, PSO cache |
| `merkle.rs` | GPU dispatch (`hash_merkle_tree_linear_threadgroup_buf_ho`) and layout conversion |
| `utils.rs` | `LinearUniforms` struct, dispatch size helpers, `POSEIDON_CONST_SIZE` |
| `buffer_pool.rs` | Persistent buffer reuse — avoids repeated Metal allocations across proof iterations |
| `tracking.rs` | `TrackedBuffer` RAII wrapper; tracks allocation/deallocation for the pool |
| `threadgroup_config.rs` | Tuning function for threadgroup sizes based on tree size |

**`plonky2/src/hash/mod.rs`** — added:

```rust
#[cfg(feature = "metal")]
pub(crate) mod metal;
```

### Hook into the Merkle tree builder

**`plonky2/src/hash/merkle_tree.rs`** — added `fill_digests_buf_metal`, called from
`fill_digests_buf_meta` when `--features metal` is set and CUDA is not:

```
fill_digests_buf_meta()
  ├── [cuda]   fill_digests_buf_gpu_cuda(...)
  ├── [metal]  fill_digests_buf_metal(...)   ← new
  └── [cpu]    fill_digests_buf(...)
```

**Routing logic inside `fill_digests_buf_metal`:**

```
H::HASHER_TYPE != Poseidon    → CPU fallback
cap_height == tree_height     → CPU fallback  (all-cap tree, no internal nodes)
tree_height < 13              → CPU fallback  (Metal launch overhead > compute benefit)
tree_height > 20              → CPU fallback  (leaves buffer too large for GPU)
otherwise                     → Metal GPU path
```

### Critical bug fix

`plonky2-fork` stores Merkle digests in **BFS / level-order** layout (root's children
at positions 0–1, leaves at the end). The original `plonky2-metal-demo` used a different
recursive in-order layout.

The conversion function `convert_linear_to_plonky2_digests` maps GPU output (linear
layout) to the fork's expected layout:

```
GPU position:  (subtree_leaves_len >> level) - 1 + node_idx   (per subtree)
CPU position:  (subtree_leaves_len >> level) - 2 + node_idx   (per subtree)
```

---

## Tests

All tests live in `plonky2/src/hash/merkle_tree.rs`, in the `mod tests` block at the
bottom of the file.

| Test | Line | Feature gate | Circuit size | Tree height |
|------|------|-------------|--------------|-------------|
| `test_merkle_trees_metal_poseidon_g64` | ~1300 | `metal` | direct Merkle (no proof) | 13 |
| `test_proof_with_metal_poseidon` | 1341 | `metal` | degree 13 (4097 gates) | 16 |
| `test_proof_cpu_poseidon` | 1376 | none | degree 13 (4097 gates) | 16 |
| `test_proof_degree17_timing` | 1411 | none | degree 17 (65537 gates) | 20 |
| `test_proof_degree19_timing` | 1445 | none | degree 19 (262145 gates) | 22 |

`test_proof_with_metal_poseidon` only compiles with `--features metal`.
The timing tests (`degree17`, `degree19`) compile in both modes so the same test binary
can serve as both the Metal and the CPU baseline.

---

## Running the Tests

> All commands should be run from the worktree root:
> `plonky2-fork/.worktrees/metal-poseidon/`

### Correctness (Merkle tree only)

```bash
cargo test --features metal --release -p plonky2 \
  test_merkle_trees_metal_poseidon_g64 -- --nocapture
```

### End-to-end proof (degree 13)

```bash
# Metal GPU path
cargo test --features metal --release -p plonky2 \
  test_proof_with_metal_poseidon -- --nocapture --test-threads=1

# CPU baseline
cargo test --release -p plonky2 \
  test_proof_cpu_poseidon -- --nocapture --test-threads=1
```

### Benchmark comparison (degree 17, max Metal range)

```bash
# Metal GPU path  — tree_height 20, Metal active
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release -p plonky2 \
  test_proof_degree17_timing -- --nocapture --test-threads=1

# CPU baseline
RUSTFLAGS="-Ctarget-cpu=native" cargo test --release -p plonky2 \
  test_proof_degree17_timing -- --nocapture --test-threads=1
```

### Benchmark comparison (degree 19)

```bash
# Metal path  — initial oracle falls back to CPU (tree_height 22 > 20),
#               but FRI round trees (height ≤ 20) get Metal acceleration
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release -p plonky2 \
  test_proof_degree19_timing -- --nocapture --test-threads=1

# CPU baseline
RUSTFLAGS="-Ctarget-cpu=native" cargo test --release -p plonky2 \
  test_proof_degree19_timing -- --nocapture --test-threads=1
```

> **Important flags**
> - `--release` — required; debug builds have different Metal buffer padding
> - `--test-threads=1` — prevents parallel test runs from competing for CPU threads
> - `RUSTFLAGS="-Ctarget-cpu=native"` — enables SIMD/AVX on the CPU path for a fair comparison

---

## Benchmark Results (Apple M-series)

| Circuit | CPU prove | Metal prove | Speedup |
|---------|-----------|-------------|---------|
| degree 13 | 284 ms | 262 ms | +8% |
| degree 17 | 5.65 s | 5.70 s | ~0% |
| degree 19 | 196.7 s | 126.6 s | **+55%** |

**Why degree 17 shows almost no gain:**
Merkle tree building is a small fraction of total prove time at this size; FFT/NTT
dominates. Metal handles all the trees (height ≤ 20) but the bottleneck is elsewhere.

**Why degree 19 is 55% faster despite the routing limit:**
During `prove()`, the FRI protocol builds a series of commitment trees at each folding
round, each half the size of the previous:

```
Initial oracle:  2^22 leaves → height 22 → CPU   (above limit)
FRI round 0:     2^21 leaves → height 21 → CPU
FRI round 1:     2^20 leaves → height 20 → Metal ✓
FRI round 2:     2^19 leaves → height 19 → Metal ✓
FRI round 3:     2^18 leaves → height 18 → Metal ✓
...
```

At degree 19, those mid-size FRI trees are large enough that GPU throughput advantage
matters. Accelerating them saves ~36% of total prove time even though the two largest
initial commitments still run on CPU.

---

## Potential Next Steps

- **Extend the routing window to tree_height 22** — would activate Metal for the degree-19
  initial oracle trees. Requires handling ~256 MB digest buffers; feasible on M-series
  unified memory but untested.
- **GPU NTT/FFT** — the remaining bottleneck for degree ≤17; already prototyped in the
  original `plonky2-metal-demo` (`metal-ntt` feature).
- **Merge to main** — open a PR from `feature/metal-poseidon`.
