# GPU Memory Bottleneck Optimization Design

**Date:** 2026-03-03
**Goal:** Improve Metal GPU Poseidon kernel performance by addressing memory bottlenecks (register spilling, occupancy, unnecessary indirection).

## Problem

Fine-grained benchmarking (3 runs × 4 configs) showed the Poseidon shader optimization (fast partial rounds + mul() fix) had **no measurable impact** on GPU absolute times (0-6%, within thermal noise). The GPU is not ALU-bound.

Analysis of the shader code identified 5 memory-related bottlenecks:

1. **Register spilling** — Poseidon + MDS uses ~130+ 32-bit register equivalents; Apple Silicon has ~32. Heavy spilling to device memory.
2. **Low occupancy (25-50%)** — 8,088-16,280 bytes threadgroup memory per threadgroup limits concurrent threadgroups per EU.
3. **Pointless shared_children cache** — Internal node kernel writes children to TG memory, barrier, reads back own data. No cross-thread sharing.
4. **8KB constant loading per dispatch** — Cooperative TG loading with 5-way branch chain, runs 14+ times per tree.
5. **Per-level dispatch overhead** — 12 separate dispatches for internal levels, tiny work at upper levels.

## Approach

4 sequential optimizations. After each: validate correctness, benchmark 3 runs, keep if faster or revert if not.

## Optimization 1: Revert Fast Partial Rounds

Revert all 3 optimization commits (`208aa365`, `9a4668f3`, `8a084c4a`). Restores:
- `goldilocks.metal` — original mul()
- Removes `poseidon_fast_partial_constants.metal`
- `poseidon_goldilocks_threadgroup.metal` — removes 6 fast partial functions
- Both kernel shader files — restores `poseidon_permute_tg_full` calls
- `utils.rs` — POSEIDON_CONST_SIZE back to 2,976 bytes

**Expected impact:** Threadgroup memory 8,088 → 2,976 bytes. Leaf kernel occupancy 50% → potentially higher. Removes branchy constant loading for 639 extra values.

## Optimization 2: Remove shared_children Indirection

In the internal node kernel of both `poseidon_merkle_hasher_linear_threadgroup.metal` and `poseidon_merkle_hasher_coalesced.metal`:

**Current (wasteful):**
```metal
// Write device → threadgroup
shared_children[shared_base + 0] = output[left_offset + 0];
// ... 8 writes total
threadgroup_barrier(mem_flags::mem_threadgroup);
// Read threadgroup → registers
p2_state[0] = Fp(shared_children[shared_base + 0]);
// ... 8 reads total
```

**New (direct):**
```metal
// Read device → registers directly
p2_state[0] = Fp(output[left_offset + 0]);
// ... 8 reads total
```

Remove the `shared_children` threadgroup allocation from internal kernels.

**Expected impact:** Eliminates 16 TG memory accesses + 1 barrier per thread. Reduces internal kernel TG memory from 16,280 → 8,088 bytes (child cache removed). Occupancy doubles.

## Optimization 3: Constants via Device Buffer

Replace cooperative threadgroup constant loading with a Metal `[[buffer]]` parameter.

**Current:** Constants loaded into threadgroup memory cooperatively at every kernel dispatch. 2,976 bytes loaded via loop with barrier. Counts against TG memory budget.

**New:** Create a single Metal buffer containing all constants (RC + MDS). Bind as `constant` address space buffer parameter to each kernel. Hardware L1 cache handles repeated access. No threadgroup memory needed for constants.

Changes:
- Shader: kernel signatures gain `constant ulong* poseidon_constants [[buffer(N)]]`
- Shader: `load_*_tg()` functions and barriers removed
- Shader: access functions read from `constant` pointer instead of `threadgroup` pointer
- Rust `runtime.rs`: create persistent constants buffer at init
- Rust `merkle.rs`: bind constants buffer to each kernel dispatch

**Expected impact:** TG memory drops to ~0 bytes for leaf/caps kernels (just stack). Internal kernel TG memory = 0 (no constants, no child cache after opt 2). Maximum occupancy.

## Optimization 4: Reduce MDS Register Pressure

Restructure `apply_mds_freq_tg` to reduce peak register usage. Currently holds `state_l[12]` + `state_h[12]` + FFT intermediates simultaneously (~130+ 32-bit registers, severe spilling).

Option A: Process low/high halves sequentially, reusing the same `state_x[12]` array.
Option B: Split into 3 groups of 4 elements, process independently.

This is the most speculative optimization. Implementation depends on results from optimizations 1-3.

## Testing Protocol (per optimization)

1. Recompile shaders: `xcrun -sdk macosx metal -c ... && xcrun -sdk macosx metallib ...`
2. Correctness tests:
   - `test_merkle_trees_metal_poseidon_g64` (heights 1-14)
   - `metal_coalesced` (heights 20-22)
   - `metal_proof` (e2e proof roundtrip)
3. Benchmark: 3 runs each of Metal + CPU (d13-d17 fast + d18), take best `prove_min`
4. Decision: keep if Metal speedup improved, revert if not

## Baseline (pre-optimization, from 3-run benchmark)

| Degree | CPU best | Metal best | Speedup |
|--------|----------|------------|---------|
| d13 | 290ms | 221ms | 1.31x |
| d14 | 580ms | 462ms | 1.26x |
| d15 | 1,169ms | 937ms | 1.25x |
| d16 | 2,411ms | 1,956ms | 1.23x |
| d17 | 4,990ms | 4,095ms | 1.22x |
| d18 | 14,894ms | 13,960ms | 1.07x |
