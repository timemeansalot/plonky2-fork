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

### Approach C: Coalesced shader (2026-03-02)

| Config | d18 prove_min | vs CPU-only |
|--------|--------------|-------------|
| CPU-only baseline | 17,777ms | — |
| **Coalesced shader (tree_height > 20)** | **15,686ms** | **-12% (faster!)** |

**Changes:** Route tree_height > 20 to `dispatch_merkle_coalesced` instead of CPU fallback.
**Observations:** Coalesced shader is significantly faster than CPU at d18. Previous claim of "10-15% slower" was incorrect or outdated — likely improved by BFS layout optimization and GPU dispatch thread changes made since that measurement.

### Approach B: Per-subtree GPU dispatch (2026-03-02)

| Config | d18 prove_min | vs CPU-only |
|--------|--------------|-------------|
| CPU-only baseline | 17,777ms | — |
| **Per-subtree dispatch (16 × tree_height=17)** | **16,618ms** | **-6.5% (faster)** |

**Changes:** Loop over 16 subtrees, dispatch each individually to `dispatch_merkle_linear_threadgroup` with cap_height=0.
**Observations:** Faster than CPU but slower than Approach C. The 16 separate GPU round-trips add overhead. High variance between iterations (16.6s best, 22.9s worst) due to memory pressure.

<!-- More results appended below -->
