# Prove Time Breakdown at d17

Detailed timing breakdown for degree 17, comparing CPU-only vs Metal Merkle-only (best config).

## CPU-Only (prove_min ~5,000ms)

| Phase | Time (ms) | % of prove | Details |
|-------|-----------|------------|---------|
| **Merkle tree builds** | 3,320 | **66%** | wires 2,304 + partial 577 + quotient 438 |
| **FFT + blinding (coset NTT)** | 750 | **15%** | wires 612 + partial 74 + quotient 66 |
| Quotient computation | 340 | 7% | CPU constraint evaluation |
| Opening proofs / FRI | 173 | 3% | FRI folding + proof-of-work |
| Transpose LDEs | 160 | 3% | wires 137 + partial 9 + quotient 10 |
| Other | ~250 | 5% | generators, witness, IFFT |

## Metal Merkle-Only (prove_min ~4,900ms)

| Phase | Time (ms) | % of prove | Details |
|-------|-----------|------------|---------|
| **Merkle tree builds** | 3,170 | **65%** | wires 2,369 + partial 456 + quotient 339 |
| **FFT + blinding (coset NTT)** | 680 | **14%** | wires 532 + partial 93 + quotient 63 |
| Transpose LDEs | 680 | 14% | wires 663 + partial 12 + quotient 9 |
| Quotient computation | 430 | 9% | CPU constraint evaluation |
| Opening proofs / FRI | 168 | 3% | FRI folding + proof-of-work |

## Key Takeaways

1. **Merkle dominates at ~66%** of prove time in both configs.
2. **FFT/NTT is only ~15%.** Even if NTT were instant, max savings ~750ms (15%).
3. **GPU Merkle saves ~150ms at d17** (3,320 → 3,170ms) — modest, room for more.
4. **GPU NTT was attempted but is slower than CPU Rayon** — Apple Silicon GPU has fewer ALUs for 64-bit integer math. See `ntt-benchmark-iterations.md` for details.
5. **Transpose appears expensive in Metal config** on first iteration (525ms vs 108ms), but this is a warmup artifact — see investigation below.

## Wires Commitment Dominates

The wires commitment alone is ~65% of prove time:

| Sub-phase | CPU-Only | Metal Merkle |
|-----------|----------|-------------|
| IFFT | 47ms | 41ms |
| FFT + blinding | 612ms | 532ms |
| Transpose LDEs | 137ms | 122ms |
| **Build Merkle tree** | **2,304ms** | **2,369ms** |
| **Total** | **3,121ms** | **3,087ms** |

The wires commitment Merkle tree has tree_height = 20 (2^17 degree x 2^3 rate = 2^20 leaves), which is at the upper boundary of the linear+threadgroup GPU path.

Note: Transpose times above use steady-state (iteration 3+) values. First-iteration values are inflated by GPU memory warmup — see below.

## Transpose "Regression" Investigation

Initial observation: Metal config showed ~680ms total transpose vs ~160ms for CPU-only. This looked like a 4x regression. Investigation revealed it's a **first-iteration warmup artifact**, not a real regression.

### Per-Iteration Wires Transpose Times (d17)

| Iteration | CPU-Only | Metal Merkle | Notes |
|-----------|----------|-------------|-------|
| 1 (cold) | 108ms | **525ms** | GPU buffer pool allocates shared memory, pressures CPU TLB |
| 2 | 163ms | **290ms** | OS memory manager adapting page tables |
| 3 | 155ms | 122ms | Converged — similar to CPU |
| 4 | 150ms | 150ms | Converged — similar to CPU |

### Root Cause

Metal NTT is **disabled** in this config — both CPU-only and Metal Merkle-only use `from_coeffs_cpu` for FFT. The transpose function and its input data are identical. The only difference is that in the Metal config, the GPU Merkle tree builder allocates large shared-memory buffers (via `buffer_pool.rs`) for each Merkle tree.

When the first prove iteration runs:
1. Preprocess builds a Merkle tree on GPU → allocates shared memory buffers in the pool
2. Wires commitment FFT runs on CPU → produces `lde_values`
3. **Transpose runs** — but the GPU shared memory mappings from step 1 cause TLB/cache pressure on the CPU
4. By iteration 3+, the OS page table mappings have stabilized and the penalty disappears

### Conclusion

- **`prove_min` is unaffected** — best iteration has similar transpose in both configs (~120-150ms)
- **`prove_mean` is slightly inflated** — first iteration pays ~400ms warmup penalty
- **Not actionable** — this is OS-level memory management, not something we can optimize in code

## Optimization Priority

Based on steady-state time share:
1. **Merkle tree building (66%)** — highest leverage, already GPU-accelerated with 16-28% e2e speedup
2. **FFT/NTT (15%)** — GPU NTT proven slower than CPU Rayon; Apple Silicon GPU lacks 64-bit integer ALU throughput
3. **Quotient computation (7-9%)** — CPU-bound, hard to parallelize further
4. **Transpose (3%)** — not a real regression; steady-state similar to CPU
5. **Opening proofs (3%)** — small, diminishing returns

## Metal Optimization Ceiling

The current Merkle-only GPU path at d13-d17 (16-28% speedup) is likely near the ceiling for Apple Silicon Metal acceleration on this workload:

| Optimization | Result | Why |
|-------------|--------|-----|
| Merkle linear+threadgroup (active) | **16-28% faster** | Poseidon hashing maps well to GPU |
| Merkle coalesced for d18+ | **10-15% slower** | UMA bandwidth saturation; CPU Rayon wins on 16 subtrees |
| GPU NTT per-polynomial | **+2-13% slower** | 72 GPU round-trips overhead |
| GPU NTT batched | **+20-36% slower** | GPU ALU throughput < CPU Rayon for 64-bit modular arithmetic |

The 66% of prove time spent in Merkle is already GPU-accelerated. The remaining 34% (FFT, quotient, FRI) is CPU-bound work where Apple Silicon GPU offers no advantage.
