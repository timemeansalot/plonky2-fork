# d18-d20 Merkle GPU Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Find the fastest GPU Merkle strategy for tree_height > 20 (d18-d20) by testing three approaches and keeping the winner.

**Architecture:** Test three strategies in `fill_digests_buf_metal`: (C) coalesced shader, (B) per-subtree GPU dispatch, (A) hybrid CPU+GPU split. Benchmark each at d18, keep the best, revert the rest.

**Tech Stack:** Rust (nightly-2024-01-16), Metal API, Apple Silicon UMA

---

### Task 1: Get CPU baseline for d18

Before changing any code, establish the CPU-only baseline at d18.

**Step 1: Run CPU-only d18 benchmark**

Run:
```bash
cd /Users/fujie/coding/cysic/20260220/plonky2-fork
RUSTFLAGS="-Ctarget-cpu=native" cargo test --release --test bench_degrees -- full_d18 --nocapture --test-threads=1
```

Record the `prove_min` value.

**Step 2: Run Metal (current config, CPU fallback at d18) benchmark**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release --test bench_degrees -- full_d18 --nocapture --test-threads=1
```

Record the `prove_min` value. This should be similar to CPU-only since d18 falls back to CPU for Merkle.

**Step 3: Create benchmark tracker**

Create `docs/d18-d20-benchmark-iterations.md` with the baseline data.

**Step 4: Commit**

```bash
git add -f docs/d18-d20-benchmark-iterations.md
git commit -m "docs: add d18-d20 benchmark baseline"
```

---

### Task 2: Approach C — Enable coalesced shader for tree_height > 20

**Files:**
- Modify: `plonky2/src/hash/merkle_tree.rs:504`
- Modify: `plonky2/src/hash/metal/gpu_thread.rs:138`

**Step 1: Change routing to use coalesced for tree_height > 20**

In `plonky2/src/hash/merkle_tree.rs`, line 504, change:

```rust
    if cap_height == tree_height || tree_height < 13 || tree_height > 20 {
```

to:

```rust
    if cap_height == tree_height || tree_height < 13 {
```

Then change the GPU dispatch section (lines 517-519) from:

```rust
    let leaves_buf = RUNTIME.wrap_or_copy(leaves_gl);
    let (gpu_digests, gpu_caps) = gpu_thread::GPU_DISPATCHER
        .dispatch_merkle_linear_threadgroup(leaves_buf, tree_height, leaf_size, cap_height);
```

to:

```rust
    let leaves_buf = RUNTIME.wrap_or_copy(leaves_gl);
    let (gpu_digests, gpu_caps) = if tree_height > 20 {
        gpu_thread::GPU_DISPATCHER
            .dispatch_merkle_coalesced(leaves_buf, tree_height, leaf_size, cap_height)
    } else {
        gpu_thread::GPU_DISPATCHER
            .dispatch_merkle_linear_threadgroup(leaves_buf, tree_height, leaf_size, cap_height)
    };
```

**Step 2: Remove `#[allow(dead_code)]` from `dispatch_merkle_coalesced`**

In `plonky2/src/hash/metal/gpu_thread.rs`, line 138, remove:
```rust
    #[allow(dead_code)]
```

**Step 3: Verify correctness**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release --test metal_coalesced -- --nocapture --test-threads=1
```

Expected: all coalesced tests pass (tree_height 21, 22).

**Step 4: Benchmark d18**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release --test bench_degrees -- full_d18 --nocapture --test-threads=1
```

Record `prove_min`. Compare with CPU baseline.

**Step 5: Record results in tracker, then REVERT changes**

Append results to `docs/d18-d20-benchmark-iterations.md`. Then revert the code changes:

```bash
git checkout -- plonky2/src/hash/merkle_tree.rs plonky2/src/hash/metal/gpu_thread.rs
```

**Step 6: Commit results only**

```bash
git add -f docs/d18-d20-benchmark-iterations.md
git commit -m "bench: approach C (coalesced) results for d18"
```

---

### Task 3: Approach B — Per-subtree GPU dispatch

**Files:**
- Modify: `plonky2/src/hash/merkle_tree.rs:504-530`

**Step 1: Add per-subtree GPU dispatch path**

In `plonky2/src/hash/merkle_tree.rs`, replace the `tree_height > 20` CPU fallback with a per-subtree GPU loop. Change line 504 from:

```rust
    if cap_height == tree_height || tree_height < 13 || tree_height > 20 {
        fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
        return;
    }
```

to:

```rust
    if cap_height == tree_height || tree_height < 13 {
        fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
        return;
    }

    // For large trees (height > 20), dispatch each subtree individually to GPU.
    // Each subtree has tree_height = (tree_height - cap_height) which is ≤ 20.
    if tree_height > 20 {
        let leaf_count = leaves.len() / leaf_size;
        let num_subtrees = 1usize << cap_height;
        let subtree_leaves_len = leaf_count >> cap_height;
        let subtree_tree_height = tree_height - cap_height;
        let subtree_digests_len = digests_buf.len() / num_subtrees;

        let leaves_gl: &[GoldilocksField] = unsafe {
            std::slice::from_raw_parts(
                leaves.as_ptr() as *const GoldilocksField,
                leaves.len(),
            )
        };

        for i in 0..num_subtrees {
            let leaf_start = i * subtree_leaves_len * leaf_size;
            let leaf_end = (i + 1) * subtree_leaves_len * leaf_size;
            let subtree_leaves_gl = &leaves_gl[leaf_start..leaf_end];

            let leaves_buf = RUNTIME.wrap_or_copy(subtree_leaves_gl);
            // cap_height=0 means GPU builds one complete subtree with no cap split
            let (gpu_digests, gpu_caps) = gpu_thread::GPU_DISPATCHER
                .dispatch_merkle_linear_threadgroup(
                    leaves_buf,
                    subtree_tree_height,
                    leaf_size,
                    0, // single subtree, no further cap splitting
                );
            track_deallocation(subtree_leaves_gl.len() * std::mem::size_of::<GoldilocksField>());

            // Copy digests for this subtree
            let dig_start = i * subtree_digests_len;
            let dig_end = (i + 1) * subtree_digests_len;
            for (dst, src) in digests_buf[dig_start..dig_end].iter_mut().zip(gpu_digests.iter()) {
                dst.write(unsafe { *(src as *const _ as *const H::Hash) });
            }

            // Copy cap for this subtree
            assert_eq!(gpu_caps.len(), 1, "Single subtree should produce 1 cap");
            cap_buf[i].write(unsafe { *(& gpu_caps[0] as *const _ as *const H::Hash) });
        }
        return;
    }
```

**Step 2: Verify correctness**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release --test metal_coalesced -- test_linear_height_20_cap_4 --nocapture --test-threads=1
```

Then run the e2e proof test at a degree that exercises tree_height > 20. Since `metal_proof` uses d13 (tree_height=16), we need to check correctness differently. Run the full d18 benchmark (it verifies the proof):

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release --test bench_degrees -- full_d18 --nocapture --test-threads=1
```

If it doesn't crash and reports `prove_min`, correctness is verified (proof was verified in the benchmark).

**Step 3: Record `prove_min`, then REVERT changes**

Append results to `docs/d18-d20-benchmark-iterations.md`. Then revert:

```bash
git checkout -- plonky2/src/hash/merkle_tree.rs
```

**Step 4: Commit results only**

```bash
git add -f docs/d18-d20-benchmark-iterations.md
git commit -m "bench: approach B (per-subtree dispatch) results for d18"
```

---

### Task 4: Approach A — Hybrid CPU+GPU split

**Files:**
- Modify: `plonky2/src/hash/merkle_tree.rs:504-530`

**Step 1: Add hybrid CPU+GPU dispatch path**

In `plonky2/src/hash/merkle_tree.rs`, replace the `tree_height > 20` CPU fallback. Change line 504 from:

```rust
    if cap_height == tree_height || tree_height < 13 || tree_height > 20 {
        fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
        return;
    }
```

to:

```rust
    if cap_height == tree_height || tree_height < 13 {
        fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
        return;
    }

    // For large trees (height > 20), use hybrid CPU+GPU:
    // Split subtrees — some to GPU (as one dispatch), rest to CPU Rayon.
    // Both run concurrently. GPU dispatch constrained to virtual tree_height ≤ 20.
    if tree_height > 20 {
        use std::thread;

        let leaf_count = leaves.len() / leaf_size;
        let num_subtrees = 1usize << cap_height;
        let subtree_leaves_len = leaf_count >> cap_height;
        let subtree_digests_len = digests_buf.len() / num_subtrees;

        // Calculate max GPU subtrees: gpu_subtrees * subtree_leaves_len must be ≤ 2^20
        let max_gpu_leaves = 1usize << 20;
        let mut gpu_subtrees = max_gpu_leaves / subtree_leaves_len;
        if gpu_subtrees > num_subtrees {
            gpu_subtrees = num_subtrees;
        }
        if gpu_subtrees == 0 {
            // Subtrees too large even individually — full CPU fallback
            fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
            return;
        }

        let cpu_subtrees = num_subtrees - gpu_subtrees;

        let leaves_gl: &[GoldilocksField] = unsafe {
            std::slice::from_raw_parts(
                leaves.as_ptr() as *const GoldilocksField,
                leaves.len(),
            )
        };

        // GPU part: first gpu_subtrees subtrees packed as one dispatch
        let gpu_leaves_count = gpu_subtrees * subtree_leaves_len;
        let gpu_leaves_elements = gpu_leaves_count * leaf_size;
        let gpu_leaves_gl = &leaves_gl[..gpu_leaves_elements];
        let gpu_tree_height = log2_strict(gpu_leaves_count);
        // cap_height for GPU: log2(gpu_subtrees) so each subtree is independent
        let gpu_cap_height = log2_strict(gpu_subtrees);

        let leaves_buf = RUNTIME.wrap_or_copy(gpu_leaves_gl);

        // Dispatch GPU (non-blocking: send job, don't wait for result yet)
        let (gpu_tx, gpu_rx) = std::sync::mpsc::channel();
        gpu_thread::GPU_DISPATCHER
            .sender
            .send(gpu_thread::GpuJob::MerkleLinearThreadgroup {
                leaves_buffer: gpu_thread::SendableBuffer(leaves_buf),
                tree_height: gpu_tree_height,
                leaf_length: leaf_size,
                cap_height: gpu_cap_height,
                reply: gpu_tx,
            })
            .expect("GPU dispatch thread terminated unexpectedly");

        // CPU part: remaining subtrees processed by Rayon (runs concurrently with GPU)
        if cpu_subtrees > 0 {
            let cpu_leaf_start = gpu_subtrees * subtree_leaves_len * leaf_size;
            let cpu_leaves = &leaves[cpu_leaf_start..];
            let cpu_dig_start = gpu_subtrees * subtree_digests_len;

            let (_, cpu_digests) = digests_buf.split_at_mut(cpu_dig_start);
            let (_, cpu_caps) = cap_buf.split_at_mut(gpu_subtrees);

            let cpu_subtree_count = cpu_subtrees;
            let cpu_digests_chunks = cpu_digests.par_chunks_exact_mut(subtree_digests_len);
            let cpu_leaves_chunks = cpu_leaves.par_chunks_exact(subtree_leaves_len * leaf_size);

            cpu_digests_chunks.zip(cpu_caps).zip(cpu_leaves_chunks).for_each(
                |((sub_digests, sub_cap), sub_leaves)| {
                    sub_cap.write(fill_subtree::<F, H>(sub_digests, sub_leaves, leaf_size));
                },
            );
        }

        // Wait for GPU result
        let (gpu_digests, gpu_caps) =
            gpu_rx.recv().expect("GPU dispatch thread dropped reply channel");
        track_deallocation(gpu_leaves_gl.len() * std::mem::size_of::<GoldilocksField>());

        // Copy GPU results into output buffers
        let gpu_total_digests = gpu_subtrees * subtree_digests_len;
        for (dst, src) in digests_buf[..gpu_total_digests].iter_mut().zip(gpu_digests.iter()) {
            dst.write(unsafe { *(src as *const _ as *const H::Hash) });
        }
        for (dst, src) in cap_buf[..gpu_subtrees].iter_mut().zip(gpu_caps.iter()) {
            dst.write(unsafe { *(src as *const _ as *const H::Hash) });
        }
        return;
    }
```

**Important:** This approach requires accessing `gpu_thread::GpuJob` and `gpu_thread::SendableBuffer` directly (currently `pub(crate)`), and the `sender` field on `GpuDispatcher` (currently private). You may need to:
- Make `GpuDispatcher.sender` `pub(crate)` in `gpu_thread.rs`
- Or add a `dispatch_merkle_linear_threadgroup_nonblocking` method that returns the `rx` channel instead of blocking

The cleaner approach is to add a non-blocking dispatch method to `GpuDispatcher`:

In `plonky2/src/hash/metal/gpu_thread.rs`, add after `dispatch_merkle_coalesced`:

```rust
    /// Non-blocking dispatch: sends job and returns the reply receiver.
    /// Caller is responsible for calling rx.recv() to get the result.
    pub(crate) fn dispatch_merkle_linear_threadgroup_async(
        &self,
        leaves_buffer: Buffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
    ) -> mpsc::Receiver<MerkleResult> {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(GpuJob::MerkleLinearThreadgroup {
                leaves_buffer: SendableBuffer(leaves_buffer),
                tree_height,
                leaf_length,
                cap_height,
                reply: tx,
            })
            .expect("GPU dispatch thread terminated unexpectedly");
        rx
    }
```

Then in `merkle_tree.rs`, use this cleaner API instead of accessing internals:

```rust
        // Dispatch GPU (non-blocking)
        let gpu_rx = gpu_thread::GPU_DISPATCHER
            .dispatch_merkle_linear_threadgroup_async(
                leaves_buf,
                gpu_tree_height,
                leaf_size,
                gpu_cap_height,
            );

        // CPU part runs here (concurrently with GPU)...

        // Wait for GPU
        let (gpu_digests, gpu_caps) = gpu_rx.recv()
            .expect("GPU dispatch thread dropped reply channel");
```

**Step 2: Verify correctness**

Run the d18 benchmark (it verifies the proof internally):
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release --test bench_degrees -- full_d18 --nocapture --test-threads=1
```

If it reports `prove_min` without crashing, correctness is verified.

**Step 3: Benchmark d18**

The run from Step 2 gives us the benchmark. Record `prove_min`.

**Step 4: (Optional) Try different GPU subtree counts**

If the default split (max GPU subtrees) isn't optimal, try hardcoding different values:
- At d18: try gpu_subtrees = 2, 4, 8
- Record each `prove_min`

**Step 5: Record results, then REVERT changes**

Append all results to `docs/d18-d20-benchmark-iterations.md`. Then revert:

```bash
git checkout -- plonky2/src/hash/merkle_tree.rs plonky2/src/hash/metal/gpu_thread.rs
```

**Step 6: Commit results only**

```bash
git add -f docs/d18-d20-benchmark-iterations.md
git commit -m "bench: approach A (hybrid CPU+GPU) results for d18"
```

---

### Task 5: Pick winner and finalize

**Step 1: Compare all results**

Review `docs/d18-d20-benchmark-iterations.md`. Identify which approach (if any) beats CPU baseline at d18.

**Step 2: If a winner exists — re-implement and commit permanently**

Re-apply the winning approach's code changes. Run correctness tests. Then also benchmark d19 and d20 to confirm it generalizes.

```bash
# Correctness
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release --test metal_coalesced -- --nocapture --test-threads=1

# d19 benchmark
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release --test bench_degrees -- full_d19 --nocapture --test-threads=1

# d20 benchmark (slow, ~10+ min)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release --test bench_degrees -- full_d20 --nocapture --test-threads=1
```

Commit the final implementation:
```bash
git add plonky2/src/hash/merkle_tree.rs plonky2/src/hash/metal/gpu_thread.rs
git commit -m "feat(metal): enable <winning approach> for d18-d20 Merkle trees"
```

**Step 3: If no winner — document and commit findings only**

Update `docs/d18-d20-benchmark-iterations.md` with conclusion. Update `docs/metal-acceleration-status.md` with findings.

```bash
git add -f docs/d18-d20-benchmark-iterations.md docs/metal-acceleration-status.md
git commit -m "docs: d18-d20 GPU Merkle approaches all slower than CPU"
```

**Step 4: Update prove-time-breakdown doc**

Add d18 timing breakdown to `docs/prove-time-breakdown-d17.md` (rename if needed) or create `docs/prove-time-breakdown-d18.md`.
