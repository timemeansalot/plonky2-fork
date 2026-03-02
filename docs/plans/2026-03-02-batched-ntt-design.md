# Batched Metal NTT Design

**Date:** 2026-03-02
**Target:** d13-d17 prove time improvement
**Approach:** Pack all polynomials into one GPU buffer, single batched dispatch

## Problem

The current Metal NTT path processes ~72 polynomials one at a time: allocate buffer, copy data, dispatch kernel, wait, read back. This per-polynomial overhead (~5-7ms each) totals ~400ms, exceeding the GPU compute savings. Result: +8.4% regression vs CPU at d17. Path is currently disabled.

## Design

### Data Flow

```
Current (disabled, per-polynomial):
  for each of 72 polys:
    alloc → copy → dispatch → wait → read → dealloc     (72 round-trips)

Proposed (batched):
  CPU: coset_shift all 72 polys (Rayon par_iter)          (~2ms)
  CPU: pack into contiguous buffer [p0|p1|...|p71]        (~1ms)
  GPU: wrap_or_copy (UMA zero-copy if aligned)            (~0ms)
  GPU: single command buffer:
    batch_bit_reverse: 72*n threads                       (1 dispatch)
    batch_butterfly × log_n stages: 72*(n/2) threads      (log_n dispatches)
  GPU: one commit + wait                                  (1 round-trip)
  CPU: slice output back into 72 Vecs                     (~1ms)
```

### Shader Changes (ntt_goldilocks.metal)

1. **New kernel: `ntt_batch_bit_reverse`** — bit-reverse permutation across all polynomials in batch. Thread `gid` computes `batch_idx = gid / n`, `elem_idx = gid % n`, swaps within its polynomial slice.

2. **Existing kernel: `ntt_batch_butterfly`** — already in shader, handles `batch_count` polynomials. No changes needed.

### Host-Side Changes (ntt.rs)

New method:
```rust
pub fn batch_coset_ntt(
    &self,
    polys: &[Vec<GoldilocksField>],
    shift: u64,
) -> Vec<Vec<GoldilocksField>>
```

Steps:
1. Apply coset shift to each poly via Rayon `par_iter` (CPU, unchanged logic)
2. Pack shifted polys into contiguous `Vec<GoldilocksField>` of size `batch_count * n`
3. Create GPU buffer via `wrap_or_copy` (UMA zero-copy) or buffer pool
4. Cache `batch_bit_reverse` and `batch_butterfly` PSOs at MetalNTT init time
5. Encode single command buffer with all stages
6. `commit()` + `wait_until_completed()`
7. Read back and split into `batch_count` separate Vecs

### Oracle.rs Integration

Replace per-polynomial `par_iter` + `coset_ntt` in `from_coeffs_metal()` with:
1. Collect all padded polynomial coefficients
2. Call `NTT_RUNTIME.batch_coset_ntt(&padded_polys, shift)`
3. Continue with existing transpose + Merkle tree build

**Fallback:** CPU path for `extended_n < 2^16` (same threshold as before).

### Buffer Management

- Use `BUFFER_POOL` from `buffer_pool.rs` for the batch buffer
- At d17: `72 * 2^20 * 8B = 576MB` — within Apple Silicon UMA budget
- Buffer pool reuses across proof iterations (size classes are power-of-2)

### Correctness & Testing Workflow

After each code change:
1. Run `test_metal_proof_roundtrip` for e2e correctness
2. Run `bench_degrees fast` (d13-d17) for performance
3. Append results to `docs/ntt-benchmark-iterations.md`
4. Compare with baseline; revert if regression

### New Tests

- `test_batch_ntt_correctness` — verify batch output matches individual NTT for various sizes/counts
- Existing `test_metal_proof_roundtrip` validates full prove/verify

### Success Criteria

- Merkle + Batched NTT at d13-d17 is faster than Merkle-only (or at worst equal)
- If not achieved, revert and keep NTT disabled

### Files to Modify

| File | Change |
|------|--------|
| `plonky2/shaders/ntt_goldilocks.metal` | Add `ntt_batch_bit_reverse` kernel |
| `plonky2/shaders/ntt_goldilocks.metallib` | Recompile |
| `plonky2/src/hash/metal/ntt.rs` | Add `batch_coset_ntt()`, cache batch PSOs |
| `plonky2/src/fri/oracle.rs` | Replace per-poly dispatch with batch call |
| `plonky2/tests/bench_degrees.rs` | Already exists (fast d13-d17) |
| `docs/ntt-benchmark-iterations.md` | Append iteration results |
