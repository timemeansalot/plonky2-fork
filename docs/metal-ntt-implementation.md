# Metal NTT Implementation

This document summarizes the current Metal NTT/LDE implementation, how it is wired into Plonky2, why prove routing is currently disabled, and how to test or benchmark it.

## Current Status

Metal NTT is implemented and correctness-tested, but it is not active in the default proving path.

The implementation exists in `plonky2/src/hash/metal/ntt.rs` and `plonky2/shaders/ntt_goldilocks.metal`. The proving integration exists in `PolynomialBatch::from_coeffs_metal()` in `plonky2/src/fri/oracle.rs`, but the call from `PolynomialBatch::from_coeffs()` is commented out.

Reason: benchmarks showed that Metal NTT made proving slower than the Merkle-only Metal path. The current best configuration is CPU Rayon FFT/NTT plus Metal Merkle hashing.

## Implementation Method

The implementation targets the Goldilocks field and uses precompiled Metal shader code embedded into the Rust binary:

- `plonky2/shaders/ntt_goldilocks.metal`
- `plonky2/shaders/ntt_goldilocks.metallib`
- `plonky2/src/hash/metal/ntt.rs`

`MetalNTT` loads the shader library with `include_bytes!`, creates a singleton `NTT_RUNTIME`, precomputes forward and inverse twiddle tables up to `MAX_LOG_N = 24`, and exposes these operations:

- `ntt(&self, coeffs)` for forward NTT
- `intt(&self, evals)` for inverse NTT
- `lde(&self, values, rate_bits)` for subgroup LDE
- `lde_onto_coset(&self, values, rate_bits)` for coset LDE
- `coset_ntt(&self, coeffs, shift)` for one polynomial
- `batch_coset_ntt(&self, polys, shift)` for many polynomials in one packed GPU buffer

The shader provides these kernels:

- `ntt_bit_reverse`
- `ntt_butterfly`
- `intt_butterfly`
- `ntt_scale`
- `ntt_butterfly_coalesced`
- `ntt_batch_butterfly`
- `ntt_batch_bit_reverse`

The basic NTT flow is:

1. Copy input Goldilocks values to a Metal buffer.
2. Run bit reversal.
3. Run one butterfly dispatch per stage.
4. For inverse NTT, run inverse butterfly stages and scale by `n^-1`.
5. Copy results back to CPU memory.

The batched NTT flow packs multiple polynomials as:

```text
[poly0 | poly1 | ... | polyN]
```

Then it dispatches `ntt_batch_bit_reverse` and `ntt_batch_butterfly` over all polynomials at once. This removes the large per-polynomial dispatch overhead from the original GPU attempt.

For coset NTT, the implementation multiplies coefficients by `shift^i` on CPU, then runs the regular NTT on the shifted coefficients:

```text
P(shift * omega^j) = NTT(c_i * shift^i)
```

For Plonky2 FRI LDE, the shift is the Goldilocks multiplicative group generator, `7`.

## Proving Integration

The natural proving integration point is `PolynomialBatch::from_coeffs()`, because that is where Plonky2 converts polynomial coefficients into LDE values before transposing them into Merkle leaves.

The implemented GPU path is `from_coeffs_metal()` in `plonky2/src/fri/oracle.rs`:

1. Check that the field layout matches `GoldilocksField`.
2. Pad each polynomial from `degree` to `degree << rate_bits`.
3. Call `NTT_RUNTIME.batch_coset_ntt(&padded_polys, shift)`.
4. Cast results back to `F`.
5. Add blinding salt columns if needed.
6. Transpose LDE values.
7. Build the Merkle tree.

The actual routing in `from_coeffs()` is disabled with comments:

```rust
// Metal batched NTT available but currently disabled:
// GPU NTT is slower than CPU Rayon at d13-d17 sizes (~72 polys of 2^17-2^20).
// Benchmarks show +20-36% regression vs Merkle-only.
// Infrastructure kept in from_coeffs_metal() for future optimization.
```

To re-enable it for experiments, uncomment the `#[cfg(feature = "metal")]` block in `PolynomialBatch::from_coeffs()`.

## Related Code

| Path | Purpose |
|------|---------|
| `plonky2/src/hash/metal/ntt.rs` | Rust Metal NTT runtime, twiddle generation, NTT/INTT/LDE APIs, tests |
| `plonky2/shaders/ntt_goldilocks.metal` | Metal kernels for bit reversal, butterfly stages, inverse scaling, batched NTT |
| `plonky2/shaders/ntt_goldilocks.metallib` | Precompiled shader library embedded by `ntt.rs` |
| `plonky2/src/fri/oracle.rs` | `from_coeffs_metal()` integration and disabled routing block |
| `docs/ntt-benchmark-iterations.md` | Benchmark history for per-polynomial and batched NTT attempts |
| `docs/metal-acceleration-status.md` | Overall Metal status and current routing decisions |
| `docs/archived/2026-03-02-batched-ntt-design.md` | Batched NTT design rationale |
| `docs/archived/2026-03-02-batched-ntt-implementation.md` | Step-by-step implementation plan |

## Benchmark Results

All times below are `prove_min` in milliseconds for d13-d17, measured over 3 iterations. See `docs/ntt-benchmark-iterations.md` for the source table.

### Initial Per-Polynomial NTT

| Degree | CPU Only | Merkle Only | Merkle + NTT |
|--------|----------|-------------|--------------|
| d13 | 277 | 227 | 231 |
| d14 | 541 | 464 | 468 |
| d15 | 1107 | 917 | 990 |
| d16 | 2284 | 1937 | 2186 |
| d17 | 5815 | 4916 | 5098 |

The per-polynomial GPU NTT path was slightly worse at small sizes and much worse by d16. The main issue was many GPU round-trips: roughly 72 polynomials meant roughly 72 dispatch/copy/wait cycles.

### Batched NTT

| Degree | Merkle Only | Merkle + Batched NTT | Delta |
|--------|-------------|----------------------|-------|
| d13 | 227 | 273 | +20% worse |
| d14 | 464 | 565 | +22% worse |
| d15 | 917 | 1189 | +30% worse |
| d16 | 1937 | 2614 | +35% worse |
| d17 | 4916 | 6710 | +36% worse |

The batched path eliminated the dispatch overhead, but still lost to CPU Rayon. At d17, batch NTT for the wires commitment alone took about 2.37s for roughly 72 polynomials of size `2^20`.

Conclusion: for this workload, Metal NTT is compute-bound on 64-bit modular arithmetic. CPU Rayon across 8-12 CPU cores is faster than the Apple Silicon GPU path.

## Correctness Tests

Run all Metal NTT unit tests:

```bash
cd /Users/fujie/coding/cysic/20260220/plonky2-fork
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  -p plonky2 --lib -- hash::metal::ntt --nocapture
```

Important tests in `plonky2/src/hash/metal/ntt.rs`:

- `test_ntt_2_16`
- `test_ntt_2_20`
- `test_intt_roundtrip_2_16`
- `test_intt_roundtrip_2_20`
- `test_lde_onto_coset_2_16_rate_2`
- `test_lde_onto_coset_2_18_rate_3`
- `test_batch_ntt_correctness`
- `test_batch_coset_ntt_correctness`

Run the end-to-end proof test after temporarily enabling NTT routing:

```bash
cd /Users/fujie/coding/cysic/20260220/plonky2-fork
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test metal_proof -- --nocapture --test-threads=1
```

## Benchmark Procedure

Use the d13-d17 fast benchmark to compare configurations:

```bash
cd /Users/fujie/coding/cysic/20260220/plonky2-fork

# CPU baseline
RUSTFLAGS="-Ctarget-cpu=native" cargo test --release \
  --test bench_degrees -- fast --nocapture --test-threads=1

# Metal path
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test bench_degrees -- fast --nocapture --test-threads=1
```

To benchmark NTT specifically:

1. Record CPU baseline without `--features metal`.
2. Record current Metal Merkle-only baseline with `--features metal` and NTT routing disabled.
3. Temporarily enable the `from_coeffs_metal()` block in `PolynomialBatch::from_coeffs()`.
4. Run the same Metal benchmark again.
5. Compare `prove_min` by degree.

Use this table format when appending results to `docs/ntt-benchmark-iterations.md`:

```markdown
| Degree | Merkle Only | Merkle + Batched NTT | Delta vs Merkle Only |
|--------|-------------|----------------------|----------------------|
| d13 | | | |
| d14 | | | |
| d15 | | | |
| d16 | | | |
| d17 | | | |
```

## Shader Rebuild

After editing `ntt_goldilocks.metal`, rebuild the Metal library:

```bash
cd /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2/shaders
xcrun -sdk macosx metal -c ntt_goldilocks.metal -o ntt_goldilocks.air
xcrun -sdk macosx metallib ntt_goldilocks.air -o ntt_goldilocks.metallib
rm ntt_goldilocks.air
```

Then rerun the NTT unit tests and the proof roundtrip test.

## Practical Guidance

Keep NTT routing disabled unless a new shader or batching strategy beats the Merkle-only baseline. The current code is useful as tested infrastructure and as a starting point for future optimization, but it should not be enabled in production proving on Apple Silicon without fresh benchmark evidence.
