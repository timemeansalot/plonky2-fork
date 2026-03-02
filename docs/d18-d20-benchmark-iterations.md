# d18-d20 Merkle GPU Optimization Benchmark Tracker

All times are `prove_min` in milliseconds, 2 iterations each.

## Baseline (2026-03-02)

| Config | d18 prove_min |
|--------|--------------|
| CPU-only (no --features metal) | **17,777ms** |
| Metal (current: CPU fallback at d18) | 19,904ms |

**Note:** Metal config is ~12% slower than CPU-only at d18 even though all d18 Merkle trees fall back to CPU (tree_height=21 > 20). The overhead is from Metal feature initialization (device, buffer pool, PSO compilation) and possible memory pressure from GPU shared memory mappings.

**Target:** Beat CPU-only baseline of 17,777ms.

## Iteration Log

<!-- Results appended after each approach test -->
