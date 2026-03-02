# Metal GPU Acceleration Status

## Active Components

| Component | Range | Status | Speedup |
|-----------|-------|--------|---------|
| Merkle — linear+threadgroup | tree_height 13..=20 | **Active** | 16-28% faster than CPU |
| Merkle — coalesced | tree_height == 21 | **Active** | 12-14% faster than CPU |
| Merkle — coalesced | tree_height >= 22 | **Disabled** (UMA bandwidth cliff, GPU 2x slower) | N/A |
| NTT/LDE (batched) | log_n + rate_bits >= 16 | **Disabled** (20-36% slower; GPU ALU < CPU Rayon for 64-bit math) | N/A |
| Merkle — CPU fallback | tree_height < 13 or >= 22, all-cap, non-Poseidon | **Active** | baseline |

Feature priority: **CUDA > Metal > CPU**. Metal activates with `--features metal` when CUDA is absent.

## Routing Logic

### Merkle Tree (`plonky2/src/hash/merkle_tree.rs:499-506`)

```
Hasher != Poseidon         → CPU
cap_height == tree_height  → CPU  (all-cap trees, no internal digests)
tree_height < 13           → CPU  (Metal dispatch overhead > compute benefit)
tree_height 13..=20        → GPU linear+threadgroup
tree_height == 21          → GPU coalesced  (d18: 12-14% faster than CPU)
tree_height >= 22          → CPU  (UMA bandwidth cliff, GPU 2x slower)
```

### NTT (`plonky2/src/fri/oracle.rs`)

```
All sizes                  → CPU  (GPU NTT disabled; batched NTT 20-36% slower than CPU Rayon)
```

Infrastructure for GPU batched NTT exists in `from_coeffs_metal()` and `ntt.rs:batch_coset_ntt()` but is commented out in `from_coeffs()`.

## Metal Module Structure (`plonky2/src/hash/metal/`)

| File | Purpose |
|------|---------|
| `mod.rs` | Module declaration, re-exports `RUNTIME` |
| `runtime.rs` | `MetalRuntime`: device init, pipeline states, buffer allocation |
| `gpu_thread.rs` | Dedicated dispatch thread via `std::thread` + `mpsc::channel` |
| `merkle.rs` | `hash_merkle_tree_linear_threadgroup_buf_ho` and `hash_merkle_tree_coalesced_buf_ho` |
| `ntt.rs` | GPU NTT/INTT/LDE with Goldilocks field arithmetic |
| `buffer_pool.rs` | Power-of-2 buffer reuse pool (2GB max) |
| `tracking.rs` | Memory allocation tracking, `TrackedBuffer` RAII wrapper |
| `threadgroup_config.rs` | Adaptive threadgroup sizing based on tree height |
| `utils.rs` | `LinearUniforms`, `CoalescedUniforms`, dispatch helpers |

## Shader Files (`plonky2/shaders/`)

| File | Status |
|------|--------|
| `poseidon_merkle_hasher_linear_threadgroup.metallib` | **Active** — embedded in `runtime.rs` |
| `poseidon_merkle_hasher_coalesced.metallib` | **Loaded but not dispatched** |
| `ntt_goldilocks.metallib` | **Active** — embedded in `ntt.rs` |
| `poseidon_merkle_hasher_linear_threadgroup.metal` | Source for linear+threadgroup shader |
| `poseidon_merkle_hasher_coalesced.metal` | Source for coalesced shader |
| `poseidon_goldilocks.metal` | Poseidon hash implementation |
| `poseidon_goldilocks_threadgroup.metal` | Threadgroup-optimized Poseidon |
| `poseidon_goldilocks_mds.metal` | MDS matrix constants |
| `ntt_goldilocks.metal` | NTT butterfly kernels |
| `goldilocks.metal` | Field arithmetic utilities |
| `u128.h.metal` | 128-bit arithmetic header |

## All Metal Tests

### Integration Tests (`plonky2/tests/`)

| File | Test | What It Exercises |
|------|------|-------------------|
| `metal_coalesced.rs` | `test_coalesced_height_21_cap_4` | Coalesced shader, 2^21 leaves |
| | `test_coalesced_height_22_cap_4` | Coalesced shader, 2^22 leaves (~256MB) |
| | `test_coalesced_height_21_cap_1` | Coalesced shader, minimal cap |
| | `test_linear_height_20_cap_4` | Linear+threadgroup, 2^20 leaves |
| `metal_proof.rs` | `test_metal_proof_roundtrip` | Full prove→verify with GPU NTT + GPU Merkle |
| `bench_degrees.rs` | `fast_d13`..`fast_d17` | Prove timing, d13-d17 (~45s) |
| | `full_d13`..`full_d20` | Prove timing, d13-d20 (~20+ min) |
| `bench_coalesced.rs` | `bench_merkle_height_20` | Merkle-only timing, linear+threadgroup |
| | `bench_merkle_height_21` | Merkle-only timing, coalesced path |
| | `bench_merkle_height_22` | Merkle-only timing, coalesced path |

### Unit Tests (in `plonky2/src/hash/metal/`)

| Module | Test | What It Exercises |
|--------|------|-------------------|
| `utils.rs` | `test_coalesced_uniforms_size` | CoalescedUniforms is 40 bytes |
| | `test_linear_uniforms_size` | LinearUniforms is 28 bytes |
| `ntt.rs` | `test_pow_mod` | Modular exponentiation |
| | `test_mod_inverse` | Modular inverse |
| | `test_primitive_root` | Primitive root computation |
| | `test_ntt_2_16` | NTT at 2^16 |
| | `test_ntt_2_20` | NTT at 2^20 |
| | `test_intt_roundtrip_2_16` | NTT→INTT roundtrip 2^16 |
| | `test_intt_roundtrip_2_20` | NTT→INTT roundtrip 2^20 |
| | `test_lde_onto_coset_2_16_rate_2` | LDE coset 2^16 rate=2 |
| | `test_lde_onto_coset_2_18_rate_3` | LDE coset 2^18 rate=3 |

### In-Crate Test (`plonky2/src/hash/merkle_tree.rs`)

| Test | What It Exercises |
|------|-------------------|
| `test_merkle_trees_metal_poseidon_g64` | Merkle correctness at heights 1-14 |

## How to Run

All commands from the repository root.

### Correctness Tests

```bash
# Merkle tree correctness (heights 1-14, unit test)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  -p plonky2 --lib -- test_merkle_trees_metal_poseidon_g64 --nocapture

# Coalesced + linear correctness (heights 20-22)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test metal_coalesced -- --nocapture --test-threads=1

# End-to-end proof roundtrip (GPU NTT + GPU Merkle)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test metal_proof -- --nocapture --test-threads=1

# NTT unit tests
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  -p plonky2 --lib -- hash::metal::ntt --nocapture
```

### Performance Benchmarks

```bash
# Fast benchmark: d13-d17 (~45 seconds)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test bench_degrees -- fast --nocapture --test-threads=1

# Full benchmark: d13-d20 (~20+ minutes)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test bench_degrees -- full --nocapture --test-threads=1

# CPU baseline (same commands without --features metal)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --release \
  --test bench_degrees -- fast --nocapture --test-threads=1

# Merkle-only benchmark (heights 20-22)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test bench_coalesced -- --nocapture --test-threads=1
```

### Shader Recompilation (after editing .metal files)

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

## Performance Results

See [metal-bfs-layout-optimization.md](metal-bfs-layout-optimization.md) for detailed benchmarks.

| Degree | CPU prove_min | Metal prove_min | Speedup | GPU Path |
|--------|--------------|-----------------|---------|----------|
| d13 | 277ms | 217ms | **1.28x** | linear+threadgroup |
| d14 | 563ms | 452ms | **1.25x** | linear+threadgroup |
| d15 | 1.11s | 931ms | **1.20x** | linear+threadgroup |
| d16 | 2.25s | 1.94s | **1.16x** | linear+threadgroup |
| d17 | 4.93s | 4.24s | **1.16x** | linear+threadgroup |
| d18 | 17.8s | 15.7s | **1.13x** | coalesced |
| d19 | 99s | — | CPU fallback | CPU (tree_height=22) |

Metal is 16-28% faster for d13-d17 (linear+threadgroup) and 12-14% faster at d18 (coalesced). At d19+ the GPU hits a UMA bandwidth cliff — CPU fallback is used.
