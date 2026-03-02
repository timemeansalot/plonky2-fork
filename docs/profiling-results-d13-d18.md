# Fine-Grained Merkle Profiling Results (d13-d18)

**Date:** 2026-03-02
**Machine:** Apple Silicon (Metal GPU), thermal conditions varied during benchmarks

## Key Finding: GPU Execute Time IS the Bottleneck

Across all degrees, the GPU Merkle path breaks down as:

| Phase | Time | % of GPU total |
|-------|------|----------------|
| Buffer alloc + zeroing | <1ms | <0.1% |
| Command encoding | <1ms | <0.1% |
| **GPU execution** | **>99%** | **dominant** |
| Result readback | 1-30ms | 0.5-5% |
| Digest copy (host) | 1-60ms | 0.5-5% |
| wrap_or_copy | <1ms | <0.1% |

There is **no optimization opportunity** in buffer management, encoding, or data transfer. The GPU kernel execution itself is the bottleneck.

## GPU Timing Breakdown (steady-state, 3rd+ iteration)

### Wires Commitment (largest tree per degree)

| Degree | tree_height | GPU execute | readback | digest_copy | total |
|--------|------------|-------------|----------|-------------|-------|
| d13 | 16 | 16ms | 0.1ms | 0.1ms | 16ms |
| d14 | 17 | 30ms | 0.5ms | 0.2ms | 31ms |
| d15 | 18 | 60ms | 0.4ms | 0.4ms | 61ms |
| d16 | 19 | 115ms | 2.2ms | 2.5ms | 120ms |
| d17 | 20 | 230ms | 4.5ms | 6ms | 241ms |
| d18 | 21 (coal.) | 486ms | 20ms | 32ms | 539ms |

GPU execution scales ~2x per degree (doubling leaves). Readback and digest_copy grow with tree size but remain small (<10% at d18).

### At d18: Readback + Digest Copy = ~10% of GPU time

At d18 (coalesced, tree_height=21), readback (20ms) + digest_copy (32ms) = 52ms out of ~539ms total (10%). This is the first degree where host-side overhead is non-trivial. At d13-d17 it's <2%.

## CPU Timing Breakdown (steady-state, 3rd+ iteration)

### Wires Commitment (largest tree per degree)

| Degree | tree_height | leaf_hash (cum.) | internal (cum.) | wall | leaf:internal ratio |
|--------|------------|-----------------|-----------------|------|---------------------|
| d13 | 16 | 140ms | 62ms | 20ms | 2.3:1 |
| d14 | 17 | 300ms | 133ms | 41ms | 2.3:1 |
| d15 | 18 | 690ms | 222ms | 82ms | 3.1:1 |
| d16 | 19 | 1,400ms | 420ms | 167ms | 3.3:1 |
| d17 | 20 | 2,800ms | 980ms | 345ms | 2.9:1 |
| d18 | 21 | 6,500ms | 1,400ms | 750ms | 4.6:1 |

**Leaf hashing dominates at ~75% of CPU Merkle time** (3:1 ratio to internal nodes).

### Parallelism Efficiency

With 16 subtrees across ~12 CPU cores:

| Degree | tree_height | cumulative thread-time | wall time | efficiency |
|--------|------------|----------------------|-----------|------------|
| d13 | 16 | 202ms | 20ms | 10x |
| d15 | 18 | 912ms | 82ms | 11x |
| d17 | 20 | 3,780ms | 345ms | 11x |
| d18 | 21 | 7,900ms | 750ms | 10.5x |

Rayon achieves ~10-11x parallelism on 12 cores (83-92% efficiency). This is near-optimal.

## GPU vs CPU Comparison (wires commitment Merkle only)

| Degree | GPU Merkle wall | CPU Merkle wall | GPU speedup |
|--------|----------------|-----------------|-------------|
| d13 (h=16) | 16ms | 20ms | 1.25x |
| d14 (h=17) | 31ms | 41ms | 1.32x |
| d15 (h=18) | 61ms | 82ms | 1.34x |
| d16 (h=19) | 120ms | 167ms | 1.39x |
| d17 (h=20) | 241ms | 345ms | 1.43x |
| d18 (h=21) | 539ms | 750ms | 1.39x |

GPU Merkle is consistently 25-43% faster than CPU Merkle for the wires commitment tree. The speedup increases with tree size up to d17, then plateaus at d18.

## Actionable Insights

### 1. No low-hanging fruit in GPU overhead

Buffer allocation, encoding, and data transfer are all <1% of GPU time. There's nothing to optimize on the host side.

### 2. GPU kernel execution is the target

If we want faster GPU Merkle, we need to optimize the Metal shader itself (Poseidon hash computation). This is a much harder optimization.

### 3. Readback + digest copy at d18

At d18, readback (20ms) + digest_copy (32ms) = 52ms = ~10% of GPU Merkle time. This could potentially be improved by:
- Using shared memory pointers directly instead of copying (requires unsafe lifetime management)
- Overlapping readback with next phase (pipelining)

### 4. CPU leaf hashing dominates (75% of CPU Merkle)

On the CPU path, leaf hashing (`hash_or_noop`) takes ~75% of time. Internal node hashing (`two_to_one`) is only ~25%. This matches expectations: leaves hash `leaf_size` field elements (8-135 values), while internal nodes hash only 2 digests (8 u64s each).

### 5. CPU parallelism is near-optimal

Rayon achieves 83-92% parallel efficiency with 16 subtrees on 12 cores. There's minimal room to improve CPU parallelism.

### 6. The real GPU advantage

The GPU's advantage comes from its ability to hash many leaves simultaneously with thousands of threads. The Poseidon GPU kernel processes all leaves in parallel within a single dispatch, achieving higher throughput than Rayon's 12-core parallelism for the same work.

## Conclusion

The current Metal GPU Merkle implementation is well-optimized at the systems level. All overhead (buffer management, encoding, data transfer) is negligible. The bottleneck is the GPU Poseidon hash kernel itself. Further improvements would require:
1. Shader-level optimization (Poseidon round constant access patterns, register usage)
2. Algorithmic changes (different hash function, batched Poseidon)
3. Hardware improvements (more GPU compute units)

For the CPU path, leaf hashing is the dominant cost, and Rayon parallelism is near-optimal. No easy wins remain.
