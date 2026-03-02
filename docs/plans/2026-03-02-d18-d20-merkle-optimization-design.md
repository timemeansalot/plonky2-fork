# d18-d20 Merkle Tree GPU Optimization Design

**Date:** 2026-03-02
**Target:** d18-d20 prove time improvement
**Approach:** Test three strategies for GPU Merkle at tree_height > 20

## Problem

Metal GPU Merkle tree hashing is disabled for tree_height > 20 (d18-d20 with rate_bits=3). The GPU is 10-15% slower than CPU Rayon at these sizes due to UMA memory bandwidth saturation. This means d18-d20 get no GPU acceleration despite Merkle being 66% of prove time.

At d18: 16 independent subtrees, each with 2^17 leaves (tree_height=17 per subtree — within GPU sweet spot). The problem is processing all 16 together creates tree_height=21.

## Three Approaches

### Approach C: Enable Coalesced Shader (quick re-test)

Route tree_height > 20 to the existing coalesced shader instead of CPU fallback.

- Infrastructure already built: `dispatch_merkle_coalesced` exists, shader compiled
- Previous data: 10-15% slower than CPU — re-test to confirm
- Change: remove `tree_height > 20` from CPU fallback condition in `fill_digests_buf_metal`

### Approach B: Per-Subtree GPU Dispatch

Loop over the 16 subtrees and dispatch each individually to `dispatch_merkle_linear_threadgroup`. Each subtree has tree_height ≤ 20, so the existing GPU kernel handles it.

- 16 separate GPU round-trips (dispatch, wait, read back)
- Expected: per-dispatch overhead (~5-7ms each ≈ 80-112ms total) may be acceptable if GPU compute savings exceed it
- Each subtree: 2^17 leaves at d18 → tree_height=17, well within GPU range

### Approach A: Hybrid CPU+GPU Split

Split 16 subtrees between GPU and CPU. Pack K subtrees into one GPU dispatch (virtual tree_height ≤ 20), Rayon processes the rest concurrently.

Subtree allocation constrained by virtual tree_height ≤ 20:

| Degree | Subtree leaves | Max GPU subtrees | Virtual tree_height | CPU subtrees |
|--------|---------------|-----------------|-------------------|-------------|
| d18 | 2^17 | 8 | 20 | 8 |
| d19 | 2^18 | 4 | 20 | 12 |
| d20 | 2^19 | 2 | 20 | 14 |

GPU dispatch is non-blocking (via existing gpu_thread). CPU Rayon runs in parallel. Total time = max(GPU_time, CPU_time) instead of CPU_time_for_all_16.

## Testing Order

1. **Approach C** (5 min) — change one routing condition, benchmark d18
2. **Approach B** (30 min) — add subtree loop, benchmark d18
3. **Approach A** (1-2 hours) — build hybrid split, benchmark d18 with K=2,4,8

## Benchmark Protocol

For each approach:
1. Verify correctness: `test_metal_proof_roundtrip` or equivalent at d18+
2. Benchmark d18 (3 iterations, `prove_min`)
3. Compare with CPU baseline d18 (~20.6s from prior benchmarks)
4. If promising, also test d19-d20
5. Record results in benchmark tracker

## Files to Modify

| Approach | File | Change |
|----------|------|--------|
| C | `plonky2/src/hash/merkle_tree.rs:504` | Route tree_height > 20 to coalesced |
| C | `plonky2/src/hash/metal/gpu_thread.rs:138` | Remove `#[allow(dead_code)]` |
| B | `plonky2/src/hash/merkle_tree.rs` | New path: loop-dispatch subtrees |
| A | `plonky2/src/hash/merkle_tree.rs` | New path: split subtrees, concurrent dispatch |
| All | `docs/` | Record results |

## Success Criteria

- Any approach faster than CPU-only at d18-d20 → keep it
- Best approach gets kept, others reverted
- If none faster → keep CPU fallback, document findings
