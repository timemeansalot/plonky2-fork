# Batched Metal NTT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace per-polynomial GPU NTT dispatch with a single batched dispatch to eliminate 71 of 72 GPU round-trips and make Metal NTT faster than CPU.

**Architecture:** Pack all ~72 polynomials into one contiguous buffer, dispatch the existing `ntt_batch_butterfly` shader kernel once per stage with `batch_count * n/2` threads, add a new `ntt_batch_bit_reverse` kernel. Cache pipeline states at init. Use buffer pool for reuse.

**Tech Stack:** Rust (nightly-2024-01-16), Metal Shading Language, Apple Metal API

---

### Task 1: Add `ntt_batch_bit_reverse` shader kernel

**Files:**
- Modify: `plonky2/shaders/ntt_goldilocks.metal:253` (append new kernel)

**Step 1: Add the batch bit-reverse kernel to the shader**

Append after the existing `ntt_batch_butterfly` kernel (line 253):

```metal
// Batch bit-reversal kernel
// Reorders elements in each polynomial according to bit-reversed indices
kernel void ntt_batch_bit_reverse(
    device ulong * data[[buffer(0)]],
    constant NTTUniforms & uniforms[[buffer(1)]],
    constant uint & batch_count[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    uint total_elements = uniforms.n * batch_count;
    if (gid >= total_elements) return;

    uint batch_idx = gid / uniforms.n;
    uint elem_idx = gid % uniforms.n;

    // Compute bit-reversed index within this polynomial
    uint rev = 0;
    uint x = elem_idx;
    for (uint i = 0; i < uniforms.log_n; i++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }

    // Only swap if elem_idx < rev (to avoid double-swapping)
    if (elem_idx < rev) {
        uint base = batch_idx * uniforms.n;
        ulong temp = data[base + elem_idx];
        data[base + elem_idx] = data[base + rev];
        data[base + rev] = temp;
    }
}
```

**Step 2: Recompile the metallib**

Run:
```bash
cd plonky2/shaders
xcrun -sdk macosx metal -c ntt_goldilocks.metal -o ntt_goldilocks.air
xcrun -sdk macosx metallib ntt_goldilocks.air -o ntt_goldilocks.metallib
rm ntt_goldilocks.air
```

Expected: no errors, new `.metallib` generated.

**Step 3: Verify existing NTT unit tests still pass**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release -p plonky2 --lib -- hash::metal::ntt --nocapture
```

Expected: all 9 NTT tests pass (shader change only added a kernel, didn't modify existing ones).

**Step 4: Commit**

```bash
git add plonky2/shaders/ntt_goldilocks.metal plonky2/shaders/ntt_goldilocks.metallib
git commit -m "feat(metal): add ntt_batch_bit_reverse shader kernel"
```

---

### Task 2: Cache batch PSOs and load batch function in MetalNTT

**Files:**
- Modify: `plonky2/src/hash/metal/ntt.rs:48-90` (MetalNTT struct and init)

**Step 1: Add batch function fields to MetalNTT struct**

In `plonky2/src/hash/metal/ntt.rs`, add two fields to the `MetalNTT` struct (after line 58, before `twiddle_factors`):

```rust
pub struct MetalNTT {
    device: Mutex<Device>,
    command_queue: CommandQueue,
    f_bit_reverse: Function,
    f_butterfly: Function,
    f_butterfly_inv: Function,
    f_scale: Function,
    f_batch_bit_reverse: Function,
    f_batch_butterfly: Function,
    twiddle_factors: Buffer,
    inv_twiddle_factors: Buffer,
    n_inverses: Buffer,
    max_log_n: usize,
}
```

**Step 2: Load the batch functions in the Lazy initializer**

In the `NTT_RUNTIME` Lazy block (around line 65-90), add after the `f_scale` line:

```rust
    let f_batch_bit_reverse = lib.get_function("ntt_batch_bit_reverse", None).unwrap();
    let f_batch_butterfly = lib.get_function("ntt_batch_butterfly", None).unwrap();
```

And add them to the struct initialization:

```rust
    MetalNTT {
        device: Mutex::new(Device::system_default().unwrap()),
        command_queue,
        f_bit_reverse,
        f_butterfly,
        f_butterfly_inv,
        f_scale,
        f_batch_bit_reverse,
        f_batch_butterfly,
        twiddle_factors: twiddles,
        inv_twiddle_factors: inv_twiddles,
        n_inverses: n_invs,
        max_log_n: MAX_LOG_N,
    }
```

**Step 3: Verify existing NTT tests still pass**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release -p plonky2 --lib -- hash::metal::ntt --nocapture
```

Expected: all 9 tests pass. We only added fields, no behavior changes.

**Step 4: Commit**

```bash
git add plonky2/src/hash/metal/ntt.rs
git commit -m "feat(metal): load batch NTT shader functions into MetalNTT"
```

---

### Task 3: Implement `batch_ntt_in_place` method

**Files:**
- Modify: `plonky2/src/hash/metal/ntt.rs` (add method to `impl MetalNTT`)

**Step 1: Write a unit test for batch NTT correctness**

Add to the `#[cfg(test)] mod tests` section at the bottom of `ntt.rs`:

```rust
    /// Test that batch NTT produces the same results as individual NTTs
    #[test]
    fn test_batch_ntt_correctness() {
        let log_n = 16;
        let n = 1usize << log_n;
        let batch_count = 8;

        // Create batch_count different polynomials
        let polys: Vec<Vec<GoldilocksField>> = (0..batch_count)
            .map(|b| {
                (0..n)
                    .map(|i| {
                        GoldilocksField::from_canonical_u64(
                            ((b * n + i) as u64) % GoldilocksField::ORDER,
                        )
                    })
                    .collect()
            })
            .collect();

        // Individual NTTs
        let individual_results: Vec<Vec<GoldilocksField>> =
            polys.iter().map(|p| NTT_RUNTIME.ntt(p)).collect();

        // Pack into contiguous buffer
        let mut packed: Vec<GoldilocksField> = Vec::with_capacity(batch_count * n);
        for poly in &polys {
            packed.extend_from_slice(poly);
        }

        // Create GPU buffer
        let buffer_size = packed.len() * std::mem::size_of::<u64>();
        let data_buffer = NTT_RUNTIME.device.lock().unwrap().new_buffer_with_data(
            packed.as_ptr() as *const _,
            buffer_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Batch NTT in place
        NTT_RUNTIME.batch_ntt_in_place(&data_buffer, n, log_n, batch_count, false);

        // Read back results
        let ptr = data_buffer.contents() as *const GoldilocksField;
        let result = unsafe { std::slice::from_raw_parts(ptr, batch_count * n) };

        // Compare each polynomial
        for b in 0..batch_count {
            for i in 0..n {
                assert_eq!(
                    result[b * n + i],
                    individual_results[b][i],
                    "Batch NTT mismatch at poly={}, index={}",
                    b,
                    i
                );
            }
        }
        println!(
            "Batch NTT: {} polys of size 2^{} all match individual NTTs",
            batch_count, log_n
        );
    }
```

**Step 2: Run the test to verify it fails**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release -p plonky2 --lib -- hash::metal::ntt::tests::test_batch_ntt_correctness --nocapture
```

Expected: FAIL — `batch_ntt_in_place` method doesn't exist yet.

**Step 3: Implement `batch_ntt_in_place`**

Add this method to `impl MetalNTT` (after the existing `ntt_in_place` method, around line 416):

```rust
    /// Perform batched NTT/INTT in place on a buffer containing `batch_count` polynomials
    /// of size `n` each, laid out contiguously: [poly0|poly1|...|polyN].
    fn batch_ntt_in_place(
        &self,
        data_buffer: &Buffer,
        n: usize,
        log_n: usize,
        batch_count: usize,
        inverse: bool,
    ) {
        let device = self.device.lock().unwrap();

        // Create pipeline states from cached functions
        let pipeline_bit_reverse = device
            .new_compute_pipeline_state_with_function(&self.f_batch_bit_reverse)
            .unwrap();

        let pipeline_butterfly = if inverse {
            // For inverse batch NTT, we still use the non-batch butterfly
            // with manual offset handling (batch_butterfly only does forward).
            // TODO: add intt_batch_butterfly shader if needed.
            // For now, fall back to per-polynomial INTT.
            panic!("Batch inverse NTT not yet implemented");
        } else {
            device
                .new_compute_pipeline_state_with_function(&self.f_batch_butterfly)
                .unwrap()
        };

        drop(device); // Release lock before command buffer operations

        let command_buffer = self.command_queue.new_command_buffer();

        // Phase 1: Batch bit-reversal permutation
        {
            let uniforms = NTTUniforms {
                n: n as u32,
                log_n: log_n as u32,
                stage: 0,
                direction: 0,
                twiddle_stride: 0,
            };
            let batch_count_u32 = batch_count as u32;

            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&pipeline_bit_reverse);
            encoder.set_buffer(0, Some(data_buffer), 0);
            encoder.set_bytes(
                1,
                std::mem::size_of::<NTTUniforms>() as u64,
                &uniforms as *const _ as *const _,
            );
            encoder.set_bytes(2, 4, &batch_count_u32 as *const _ as *const _);

            let total_elements = batch_count * n;
            let threads_per_group = pipeline_bit_reverse.thread_execution_width() as usize;
            let num_groups = (total_elements + threads_per_group - 1) / threads_per_group;
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: num_groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_group as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        }

        // Phase 2: Batch butterfly stages
        let twiddle_stride = (1 << self.max_log_n) / n;

        for stage in 0..log_n {
            let uniforms = NTTUniforms {
                n: n as u32,
                log_n: log_n as u32,
                stage: stage as u32,
                direction: 0,
                twiddle_stride: twiddle_stride as u32,
            };
            let batch_count_u32 = batch_count as u32;

            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&pipeline_butterfly);
            encoder.set_buffer(0, Some(data_buffer), 0);
            encoder.set_buffer(1, Some(&self.twiddle_factors), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<NTTUniforms>() as u64,
                &uniforms as *const _ as *const _,
            );
            encoder.set_bytes(3, 4, &batch_count_u32 as *const _ as *const _);

            let total_butterflies = batch_count * (n / 2);
            let threads_per_group = pipeline_butterfly.thread_execution_width() as usize;
            let num_groups = (total_butterflies + threads_per_group - 1) / threads_per_group;
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: num_groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_group as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
```

**Step 4: Run the test to verify it passes**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release -p plonky2 --lib -- hash::metal::ntt::tests::test_batch_ntt_correctness --nocapture
```

Expected: PASS — batch NTT output matches individual NTTs for all 8 polynomials.

**Step 5: Commit**

```bash
git add plonky2/src/hash/metal/ntt.rs
git commit -m "feat(metal): implement batch_ntt_in_place for batched GPU dispatch"
```

---

### Task 4: Implement `batch_coset_ntt` public method

**Files:**
- Modify: `plonky2/src/hash/metal/ntt.rs` (add method to `impl MetalNTT`)

**Step 1: Write a unit test for batch coset NTT**

Add to the `#[cfg(test)] mod tests` section:

```rust
    /// Test that batch_coset_ntt matches individual coset_ntt calls
    #[test]
    fn test_batch_coset_ntt_correctness() {
        let log_n = 16;
        let n = 1usize << log_n;
        let batch_count = 4;
        let shift = GOLDILOCKS_COSET_SHIFT;

        let polys: Vec<Vec<GoldilocksField>> = (0..batch_count)
            .map(|b| {
                (0..n)
                    .map(|i| {
                        GoldilocksField::from_canonical_u64(
                            ((b * 1000 + i) as u64) % GoldilocksField::ORDER,
                        )
                    })
                    .collect()
            })
            .collect();

        // Individual coset NTTs
        let individual_results: Vec<Vec<GoldilocksField>> = polys
            .iter()
            .map(|p| NTT_RUNTIME.coset_ntt(p, shift))
            .collect();

        // Batch coset NTT
        let batch_results = NTT_RUNTIME.batch_coset_ntt(&polys, shift);

        assert_eq!(batch_results.len(), batch_count);
        for b in 0..batch_count {
            assert_eq!(batch_results[b].len(), n);
            for i in 0..n {
                assert_eq!(
                    batch_results[b][i],
                    individual_results[b][i],
                    "Batch coset NTT mismatch at poly={}, index={}",
                    b,
                    i
                );
            }
        }
        println!(
            "Batch coset NTT: {} polys of size 2^{} all match individual coset NTTs",
            batch_count, log_n
        );
    }
```

**Step 2: Run test to verify it fails**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release -p plonky2 --lib -- hash::metal::ntt::tests::test_batch_coset_ntt_correctness --nocapture
```

Expected: FAIL — `batch_coset_ntt` method doesn't exist yet.

**Step 3: Implement `batch_coset_ntt`**

Add this public method to `impl MetalNTT` (after `coset_ntt`, around line 494):

```rust
    /// Batched coset NTT: evaluates multiple polynomials on coset shift*H.
    ///
    /// Applies coset shift on CPU (Rayon parallel), packs all polynomials into a
    /// single contiguous GPU buffer, and dispatches one batched NTT.
    /// This eliminates N-1 GPU round-trips compared to calling coset_ntt N times.
    pub fn batch_coset_ntt(
        &self,
        polys: &[Vec<GoldilocksField>],
        shift: u64,
    ) -> Vec<Vec<GoldilocksField>> {
        use plonky2_maybe_rayon::*;

        let batch_count = polys.len();
        if batch_count == 0 {
            return Vec::new();
        }

        let n = polys[0].len();
        assert!(n.is_power_of_two(), "Polynomial size must be power of 2");
        let log_n = n.trailing_zeros() as usize;
        assert!(log_n <= self.max_log_n, "NTT size exceeds maximum supported");

        // Step 1: Apply coset shift to each polynomial (CPU, Rayon parallel)
        let shifted_polys: Vec<Vec<GoldilocksField>> = polys
            .par_iter()
            .map(|coeffs| {
                assert_eq!(coeffs.len(), n, "All polynomials must be same size");
                let mut shifted = Vec::with_capacity(n);
                let mut shift_pow = 1u64;
                for &c in coeffs {
                    let s = mul_mod(c.0, shift_pow, GOLDILOCKS_PRIME);
                    shifted.push(GoldilocksField::from_canonical_u64(s));
                    shift_pow = mul_mod(shift_pow, shift, GOLDILOCKS_PRIME);
                }
                shifted
            })
            .collect();

        // Step 2: Pack into contiguous buffer [poly0|poly1|...|polyN]
        let total_elements = batch_count * n;
        let mut packed = Vec::with_capacity(total_elements);
        for poly in &shifted_polys {
            packed.extend_from_slice(poly);
        }

        // Step 3: Create GPU buffer
        let buffer_size = total_elements * std::mem::size_of::<u64>();
        track_allocation(buffer_size);
        let data_buffer = self.device.lock().unwrap().new_buffer_with_data(
            packed.as_ptr() as *const _,
            buffer_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Step 4: Batched NTT in place
        self.batch_ntt_in_place(&data_buffer, n, log_n, batch_count, false);

        // Step 5: Read back and split into individual results
        let ptr = data_buffer.contents() as *const GoldilocksField;
        let all_results = unsafe { std::slice::from_raw_parts(ptr, total_elements) };

        let results: Vec<Vec<GoldilocksField>> = (0..batch_count)
            .map(|b| all_results[b * n..(b + 1) * n].to_vec())
            .collect();

        track_deallocation(buffer_size);
        results
    }
```

**Step 4: Run test to verify it passes**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release -p plonky2 --lib -- hash::metal::ntt::tests::test_batch_coset_ntt_correctness --nocapture
```

Expected: PASS.

**Step 5: Commit**

```bash
git add plonky2/src/hash/metal/ntt.rs
git commit -m "feat(metal): implement batch_coset_ntt for batched polynomial evaluation"
```

---

### Task 5: Integrate batched NTT into `from_coeffs_metal`

**Files:**
- Modify: `plonky2/src/fri/oracle.rs:168-252` (replace per-poly dispatch)
- Modify: `plonky2/src/fri/oracle.rs:254-277` (enable Metal NTT in `from_coeffs`)

**Step 1: Rewrite `from_coeffs_metal` to use batch_coset_ntt**

Replace the body of `from_coeffs_metal` (lines 168-252) with:

```rust
    #[cfg(feature = "metal")]
    fn from_coeffs_metal(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        _fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        use plonky2_field::goldilocks_field::GoldilocksField;
        use plonky2_field::types::Field as FieldTrait;
        use plonky2_field::types::PrimeField64;
        use crate::hash::metal::ntt::NTT_RUNTIME;

        let degree = polynomials[0].len();

        // Verify F is GoldilocksField (size + alignment check)
        let is_goldilocks = std::mem::size_of::<F>() == std::mem::size_of::<GoldilocksField>()
            && std::mem::align_of::<F>() == std::mem::align_of::<GoldilocksField>();

        if !is_goldilocks {
            return Self::from_coeffs_cpu(
                polynomials, rate_bits, blinding, cap_height, timing, _fft_root_table,
            );
        }

        let salt_size = if blinding { SALT_SIZE } else { 0 };
        let extended_n = degree << rate_bits;
        let shift = GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR.to_canonical_u64();

        // Prepare padded polynomials for batch NTT
        let padded_polys: Vec<Vec<GoldilocksField>> = timed!(
            timing,
            "Metal LDE prep",
            polynomials
                .par_iter()
                .map(|p| {
                    assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                    let coeffs_gl: &[GoldilocksField] = unsafe {
                        std::slice::from_raw_parts(
                            p.coeffs.as_ptr() as *const GoldilocksField,
                            degree,
                        )
                    };
                    let mut padded = vec![GoldilocksField::ZERO; extended_n];
                    padded[..degree].copy_from_slice(coeffs_gl);
                    padded
                })
                .collect()
        );

        // Batched coset NTT: single GPU dispatch for all polynomials
        let batch_results_gl: Vec<Vec<GoldilocksField>> = timed!(
            timing,
            "Metal batch NTT",
            NTT_RUNTIME.batch_coset_ntt(&padded_polys, shift)
        );

        // Cast results back to F and add salt columns
        let lde_values: Vec<Vec<F>> = batch_results_gl
            .into_iter()
            .map(|result_gl| unsafe {
                let ptr = result_gl.as_ptr() as *const F;
                std::slice::from_raw_parts(ptr, result_gl.len()).to_vec()
            })
            .chain(
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(extended_n)),
            )
            .collect();

        // Transpose + bit-reverse + Merkle tree (same as CPU path)
        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        reverse_index_bits_in_place(&mut leaves);
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new_from_2d(leaves, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }
```

**Step 2: Enable Metal NTT in `from_coeffs`**

Replace the `from_coeffs` body (lines 254-277) to call `from_coeffs_metal`:

```rust
    #[cfg(not(feature = "cuda"))]
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        #[cfg(feature = "metal")]
        {
            let degree = polynomials[0].len();
            let log_n = log2_strict(degree);
            // Use Metal NTT for sizes where GPU batching outperforms CPU
            if log_n + rate_bits >= 16 {
                return Self::from_coeffs_metal(
                    polynomials,
                    rate_bits,
                    blinding,
                    cap_height,
                    timing,
                    fft_root_table,
                );
            }
        }
        Self::from_coeffs_cpu(
            polynomials,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }
```

**Step 3: Verify e2e correctness**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release --test metal_proof -- --nocapture --test-threads=1
```

Expected: `test_metal_proof_roundtrip` PASS (full prove + verify with batched NTT).

**Step 4: Commit**

```bash
git add plonky2/src/fri/oracle.rs
git commit -m "feat(metal): integrate batched NTT into from_coeffs_metal"
```

---

### Task 6: Benchmark and record results

**Files:**
- Modify: `docs/ntt-benchmark-iterations.md` (append results)

**Step 1: Run fast benchmark (d13-d17)**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release --test bench_degrees -- fast --nocapture --test-threads=1
```

Record the `prove_min` values for d13-d17.

**Step 2: Append results to iteration tracker**

Add a new section to `docs/ntt-benchmark-iterations.md`:

```markdown
### Iteration 1: Batched single-buffer NTT (2026-03-02)

| Degree | Merkle Only | Merkle + Batched NTT | Delta vs Merkle Only |
|--------|-------------|---------------------|---------------------|
| d13    | 227         | <measured>          | <computed>          |
| d14    | 464         | <measured>          | <computed>          |
| d15    | 917         | <measured>          | <computed>          |
| d16    | 1937        | <measured>          | <computed>          |
| d17    | 4916        | <measured>          | <computed>          |

**E2E correctness:** PASS (test_metal_proof_roundtrip)
**Changes:** Batched NTT dispatch — all polys packed into single GPU buffer, one command buffer
**Observations:** <what we learned>
```

**Step 3: Evaluate results and decide**

- If Merkle + Batched NTT is **faster** than Merkle Only at d13-d17: keep the changes.
- If Merkle + Batched NTT is **slower**: revert Task 5 Step 2 (set `from_coeffs` back to `from_coeffs_cpu`), keep the batch infrastructure for future use.

**Step 4: Commit results**

```bash
git add docs/ntt-benchmark-iterations.md
git commit -m "bench(metal): record batched NTT iteration 1 results"
```

---

### Task 7 (conditional): Update benchmark chart

Only if Task 6 shows improvement.

**Files:**
- Modify: `docs/bench_comparison.py` (add Batched NTT data)
- Modify: `docs/bench_comparison.png` (regenerate)

**Step 1: Update chart with new data**

Add `batched_ntt` array to `bench_comparison.py` with the measured values and regenerate.

**Step 2: Commit**

```bash
git add -f docs/bench_comparison.py docs/bench_comparison.png
git commit -m "docs: update benchmark chart with batched NTT results"
```
