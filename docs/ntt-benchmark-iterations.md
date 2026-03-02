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

Results will be appended here after each optimization attempt.

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
