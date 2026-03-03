# Design: Metal Poseidon Migration into plonky2-fork

**Date**: 2026-02-25
**Status**: Approved
**Scope**: Migrate Metal-accelerated Poseidon/Merkle tree building from `plonky2-metal-demo` into `plonky2-fork/plonky2/`, gated by a `metal` feature flag following the same pattern as the existing `cuda` feature.

---

## Goal

Enable callers of `plonky2-fork` to use Metal GPU acceleration for Poseidon Merkle tree construction on Apple Silicon, speeding up the Plonky2 proof generation pipeline.

**In scope**: Merkle tree / Poseidon hashing only (not NTT/LDE).
**Out of scope**: NTT, LDE, or any non-Poseidon hash acceleration.

---

## Approach

Port the `metal-bandwidth-opt` linear+threadgroup path from `plonky2-metal-demo` directly into the `plonky2` sub-crate. Use a single `metal` feature flag. Do not create a separate sub-crate.

Expected speedup: **1.3–1.45×** for trees with 2^13–2^20 leaves (the range that matters most for typical proof sizes).

---

## Section 1: Architecture & File Structure

### New directory: `plonky2-fork/plonky2/src/hash/metal/`

All files are compiled only when `#[cfg(feature = "metal")]`.

| File | Source | Purpose |
|------|--------|---------|
| `mod.rs` | new | Module re-exports + `fill_digests_buf_metal` entry point |
| `runtime.rs` | copied from `plonky2-metal-demo/src/gpu/metal/runtime.rs` | Metal device/queue/pipeline init; embeds `.metallib` via `include_bytes!()` |
| `buffer_pool.rs` | copied from demo | Persistent buffer reuse across proof iterations |
| `tracking.rs` | copied from demo | `TrackedBuffer` RAII for allocation tracking |
| `threadgroup_config.rs` | copied from demo | Thread size selection per tree height |
| `merkle.rs` | trimmed from `merkle_linear_threadgroup.rs` | GPU dispatch (sync path only; drop async `submit_*` API) |
| `utils.rs` | trimmed from demo's `utils.rs` | `LinearUniforms`, `from_buf_raw`, `get_size_for_count` |

### Shaders: `plonky2-fork/plonky2/shaders/`

Copy the following files from `plonky2-metal-demo/shaders/`:

- `poseidon_merkle_hasher_linear_threadgroup.metal` (source, for reference)
- `poseidon_merkle_hasher_linear_threadgroup.metallib` (pre-compiled, embedded at build time)
- `poseidon_goldilocks.metal`
- `poseidon_goldilocks_mds.metal`
- `poseidon_goldilocks_threadgroup.metal`
- `goldilocks.metal`
- `u128.h.metal`

The `.metallib` is embedded in `runtime.rs` via `include_bytes!()`. No runtime shader compilation. No `build.rs` changes needed.

---

## Section 2: Integration in `merkle_tree.rs`

### Dispatch priority: CUDA > Metal > CPU

Replace the existing two-variant `fill_digests_buf_meta` with three variants:

```rust
// CUDA wins if both features are enabled
#[cfg(feature = "cuda")]
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
    digests_buf, cap_buf, leaves, leaf_size, cap_height
) {
    fill_digests_buf_gpu(...)
}

// Metal when CUDA not present
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
    digests_buf, cap_buf, leaves, leaf_size, cap_height
) {
    fill_digests_buf_metal(...)
}

// CPU fallback
#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
    digests_buf, cap_buf, leaves, leaf_size, cap_height
) {
    fill_digests_buf(...)
}
```

### `fill_digests_buf_metal` routing logic

```
if H::HASHER_TYPE != HasherType::Poseidon  → CPU  (shader handles Poseidon/Goldilocks only)
if leaf_count < 2^13                        → CPU  (Metal dispatch overhead > benefit)
if leaf_count >= 2^21                       → CPU  (bandwidth saturation on M4)
otherwise                                   → Metal linear+threadgroup
```

### Type casting

The Metal function returns `Vec<HashOut<GoldilocksField>>`. When `H::HASHER_TYPE == HasherType::Poseidon`, `H::Hash = HashOut<GoldilocksField>`. Results are written into `digests_buf` / `cap_buf` via unsafe pointer cast, following the same pattern used by the CUDA path.

### Other changes

- `hash/mod.rs`: add `#[cfg(feature = "metal")] mod metal;`
- `Cargo.toml`: add `metal = ["dep:metal"]` feature and `metal = { version = "0.26.0", optional = true }` dependency (macOS only via `#[cfg(target_os = "macos")]`)

---

## Section 3: Data Flow, Error Handling & Testing

### Data flow

```
leaves: &Vec<F>  (flat 1D, u64 values)
    │
    ├─ alloc Metal buffer, copy leaves as raw u64
    │
    └─► hash_merkle_tree_linear_threadgroup_buf_ho(
              buffer, tree_height, leaf_size, cap_height
          )
              │
              └─► (Vec<HashOut<GoldilocksField>>, Vec<HashOut<GoldilocksField>>)
                        │
                        └─► unsafe cast → write into digests_buf, cap_buf
```

The flat 1D `Vec<F>` maps directly to the Metal buffer as raw `u64` — no reshape to `Vec<Vec<F>>` needed. `tree_height = log2_strict(leaf_count)`.

### Error handling

- Panic on invalid inputs (non-power-of-two leaf count, oversized tree) — same as the demo.
- The routing guards above ensure the GPU path only sees well-formed, in-range inputs; edge cases fall through to CPU.

### Testing

1. Run existing `plonky2-fork` test suite with `--features metal` — all tests must pass (proof correctness is the primary criterion).
2. Add one Metal-specific integration test in `merkle_tree.rs`: build a tree at 2^14 leaves with the `metal` feature and verify digests match the CPU result.

---

## File Change Summary

| File | Change |
|------|--------|
| `plonky2/src/hash/metal/` | New directory with 7 modules |
| `plonky2/shaders/` | New directory with Metal shader files |
| `plonky2/src/hash/mod.rs` | Add `#[cfg(feature = "metal")] mod metal;` |
| `plonky2/src/hash/merkle_tree.rs` | Add Metal variant of `fill_digests_buf_meta` + `fill_digests_buf_metal` |
| `plonky2/Cargo.toml` | Add `metal` feature + `metal` crate dependency |
