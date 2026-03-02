# Fine-Grained Profiling for d13-d18 Optimization

**Date:** 2026-03-02
**Goal:** Add detailed timing instrumentation inside Merkle tree building to find further optimization opportunities at d13-d18.

## Problem

The existing `timing` feature shows phase-level breakdown (Merkle=66%, FFT=15%, etc.) but treats "build Merkle tree" as a black box. We need function-level granularity inside the Merkle hot path to identify what's actually slow and where to optimize next.

## Approach

Add `#[cfg(feature = "timing")]`-gated `Instant`-based timing inside Merkle tree internals, printing to stderr. Use `eprintln!` directly rather than threading `TimingTree` through Metal code.

## Instrumentation Points

### GPU Merkle Path (`fill_digests_buf_metal` in `merkle_tree.rs`)

```
build Merkle tree (GPU)
├── buffer alloc + copy       (wrap_or_copy)
├── GPU dispatch + wait       (dispatch_merkle_* → channel recv)
└── digest conversion         (copy GPU results to plonky2 layout)
```

### CPU Merkle Path (`fill_digests_buf` in `merkle_tree.rs`)

```
build Merkle tree (CPU)
├── leaf hashing              (par_chunks hash_or_noop)
└── internal node hashing     (level-by-level fill_subtree)
```

### GPU Dispatch Thread (`gpu_thread.rs`, lower priority)

```
GPU dispatch internals
├── command encode
├── GPU execution             (waitUntilCompleted)
└── result readback
```

## Files to Modify

| File | Change |
|------|--------|
| `plonky2/src/hash/merkle_tree.rs:479-540` | Add timing in `fill_digests_buf_metal` around wrap_or_copy, dispatch, digest copy |
| `plonky2/src/hash/merkle_tree.rs:197-260` | Add timing in `fill_digests_buf` around leaf hashing vs internal node hashing |
| `plonky2/src/hash/metal/gpu_thread.rs` | Add timing inside dispatch handler around encode, execute, readback |

## How to Run

```bash
# Detailed breakdown at d17 (Metal)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal,timing --release \
  --test bench_degrees -- fast_d17 --nocapture --test-threads=1

# Compare CPU-only
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features timing --release \
  --test bench_degrees -- fast_d17 --nocapture --test-threads=1
```

## Success Criteria

- Timing output shows sub-phase breakdown inside every Merkle tree build
- No performance impact when `timing` feature is disabled
- Data identifies at least one actionable optimization target
