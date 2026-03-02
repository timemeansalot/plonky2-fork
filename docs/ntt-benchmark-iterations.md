# NTT Optimization Benchmark Tracker

Baseline and iteration results for Metal NTT optimization.
All times are `prove_min` in milliseconds, d13-d17, 3 iterations each.

## Baseline (2026-03-02)

| Degree | CPU Only | Merkle Only | Merkle + NTT (per-poly) |
|--------|----------|-------------|------------------------|
| d13    | 277      | 227         | 231                    |
| d14    | 541      | 464         | 468                    |
| d15    | 1107     | 917         | 990                    |
| d16    | 2284     | 1937        | 2186                   |
| d17    | 5815     | 4916        | 5098                   |

**Observations:**
- Merkle-only is the best configuration: 17-22% faster than CPU
- Adding per-polynomial NTT makes it worse: ~2x slower LDE offsets Merkle gains
- NTT regression is due to per-polynomial buffer alloc/copy/wait overhead (72 GPU round-trips)

## Iteration Log

Results appended after each optimization attempt.

### Iteration 1: Batched single-buffer NTT (2026-03-02)

| Degree | Merkle Only | Merkle + Batched NTT | Delta vs Merkle Only |
|--------|-------------|---------------------|---------------------|
| d13    | 227         | 273                 | +20% (worse)        |
| d14    | 464         | 565                 | +22% (worse)        |
| d15    | 917         | 1189                | +30% (worse)        |
| d16    | 1937        | 2614                | +35% (worse)        |
| d17    | 4916        | 6710                | +36% (worse)        |

**E2E correctness:** PASS (test_metal_proof_roundtrip)
**Changes:** Batched NTT — all ~72 polys packed into single contiguous GPU buffer, one command buffer with batch_bit_reverse + log_n batch_butterfly dispatches, single commit+wait.
**Observations:**
- GPU NTT is fundamentally slower than CPU Rayon parallel FFT at these sizes
- At d17, batch NTT for wires commitment alone takes 2.37s (72 polys × 2^20)
- The bottleneck is NOT dispatch overhead (we eliminated that) — it's raw GPU compute throughput
- Apple Silicon GPU has fewer ALUs than 8-12 CPU cores running Rayon; NTT is compute-bound, not memory-bound
- **Decision: revert routing, keep infrastructure disabled.** Merkle-only remains the best configuration.

<!-- TEMPLATE for new iterations:
### Iteration N: <description> (YYYY-MM-DD)

| Degree | Merkle Only | Merkle + Batched NTT | Delta vs Merkle Only |
|--------|-------------|---------------------|---------------------|
| d13    |             |                     |                     |
| d14    |             |                     |                     |
| d15    |             |                     |                     |
| d16    |             |                     |                     |
| d17    |             |                     |                     |

**E2E correctness:** PASS/FAIL (test_metal_proof_roundtrip)
**Changes:** <brief description>
**Observations:** <what we learned>
-->
