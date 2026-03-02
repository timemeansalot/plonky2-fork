# Fine-Grained Merkle Profiling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add detailed timing instrumentation inside Merkle tree building to identify further optimization opportunities at d13-d18.

**Architecture:** Add `#[cfg(feature = "timing")]`-gated `std::time::Instant`-based timing inside 3 locations: GPU host-side routing, GPU device-side internals, and CPU Merkle path. Output via `eprintln!` with structured prefixes for easy parsing.

**Tech Stack:** Rust, `std::time::Instant`, `#[cfg(feature = "timing")]`, `std::sync::atomic::AtomicU64`

---

### Task 1: Add timing to GPU host-side Merkle path

**Files:**
- Modify: `plonky2/src/hash/merkle_tree.rs:479-537`

This instruments `fill_digests_buf_metal` to measure 3 phases:
1. `wrap_or_copy` — buffer allocation + data transfer to GPU-visible memory
2. GPU dispatch + wait — total round-trip through the dispatch thread
3. Digest copy — copying GPU results into the plonky2 digests/caps buffers

**Step 1: Add timing instrumentation**

In `fill_digests_buf_metal`, wrap the 3 phases with `Instant` timing. Add this code after the `leaves_gl` cast (line 514) and before the `track_deallocation` call (line 527):

```rust
// After line 514 (leaves_gl cast), before line 519 (wrap_or_copy):

#[cfg(feature = "timing")]
let t0 = std::time::Instant::now();

let leaves_buf = RUNTIME.wrap_or_copy(leaves_gl);

#[cfg(feature = "timing")]
let t1 = std::time::Instant::now();

let (gpu_digests, gpu_caps) = if tree_height > 20 {
    gpu_thread::GPU_DISPATCHER
        .dispatch_merkle_coalesced(leaves_buf, tree_height, leaf_size, cap_height)
} else {
    gpu_thread::GPU_DISPATCHER
        .dispatch_merkle_linear_threadgroup(leaves_buf, tree_height, leaf_size, cap_height)
};

#[cfg(feature = "timing")]
let t2 = std::time::Instant::now();

track_deallocation(leaves_gl.len() * std::mem::size_of::<GoldilocksField>());

// ... existing digest copy code ...
for (dst, src) in digests_buf.iter_mut().zip(gpu_digests.iter()) {
    dst.write(unsafe { *(src as *const _ as *const H::Hash) });
}
for (dst, src) in cap_buf.iter_mut().zip(gpu_caps.iter()) {
    dst.write(unsafe { *(src as *const _ as *const H::Hash) });
}

#[cfg(feature = "timing")]
{
    let t3 = std::time::Instant::now();
    let path = if tree_height > 20 { "coalesced" } else { "linear_tg" };
    eprintln!(
        "[merkle-gpu h={} {}] wrap_or_copy={:.1}ms gpu_dispatch={:.1}ms digest_copy={:.1}ms total={:.1}ms",
        tree_height, path,
        t1.duration_since(t0).as_secs_f64() * 1000.0,
        t2.duration_since(t1).as_secs_f64() * 1000.0,
        t3.duration_since(t2).as_secs_f64() * 1000.0,
        t3.duration_since(t0).as_secs_f64() * 1000.0,
    );
}
```

**Step 2: Verify it compiles and runs**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal,timing --release \
  --test bench_degrees -- fast_d17 --nocapture --test-threads=1 2>&1 | grep -E "(merkle-gpu|degree)"
```
Expected: Lines like `[merkle-gpu h=20 linear_tg] wrap_or_copy=X.Xms gpu_dispatch=X.Xms digest_copy=X.Xms total=X.Xms`

Also verify no timing output without the feature:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test bench_degrees -- fast_d13 --nocapture --test-threads=1 2>&1 | grep "merkle-gpu"
```
Expected: No output (timing gated behind feature flag).

**Step 3: Commit**

```bash
git add plonky2/src/hash/merkle_tree.rs
git commit -m "feat(timing): add host-side GPU Merkle profiling"
```

---

### Task 2: Add timing to GPU device-side Merkle internals

**Files:**
- Modify: `plonky2/src/hash/metal/merkle.rs`

This instruments both `hash_merkle_tree_linear_threadgroup_buf_ho` and `hash_merkle_tree_coalesced_buf_ho` to measure 4 phases inside the GPU thread:
1. Buffer allocation (digests + caps + zeroing)
2. Command encoding (all encoder setup)
3. GPU execution (commit + waitUntilCompleted)
4. Result readback (from_buf_raw + pool return)

**Step 1: Add timing to `hash_merkle_tree_linear_threadgroup_buf_ho`**

At the top of the function (after the asserts, line ~29), add:
```rust
#[cfg(feature = "timing")]
let gt0 = std::time::Instant::now();
```

After caps buffer zeroing (after line ~61), before pipeline states:
```rust
#[cfg(feature = "timing")]
let gt1 = std::time::Instant::now();
```

After caps encoding `end_encoding()` (line ~204), before `command_buffer.commit()`:
```rust
#[cfg(feature = "timing")]
let gt2 = std::time::Instant::now();
```

After `command_buffer.wait_until_completed()` (line ~209), before reading results:
```rust
#[cfg(feature = "timing")]
let gt3 = std::time::Instant::now();
```

Before the final return `(digests, caps)` (line ~223):
```rust
#[cfg(feature = "timing")]
{
    let gt4 = std::time::Instant::now();
    eprintln!(
        "[merkle-gpu-inner h={} linear_tg] alloc={:.1}ms encode={:.1}ms execute={:.1}ms readback={:.1}ms total={:.1}ms",
        tree_height,
        gt1.duration_since(gt0).as_secs_f64() * 1000.0,
        gt2.duration_since(gt1).as_secs_f64() * 1000.0,
        gt3.duration_since(gt2).as_secs_f64() * 1000.0,
        gt4.duration_since(gt3).as_secs_f64() * 1000.0,
        gt4.duration_since(gt0).as_secs_f64() * 1000.0,
    );
}
```

**Step 2: Add identical timing to `hash_merkle_tree_coalesced_buf_ho`**

Same pattern — 4 timing points:
- `gt0`: start of function (after asserts)
- `gt1`: after buffer allocation + zeroing (after line ~274)
- `gt2`: after all encoding, before `command_buffer.commit()` (after line ~443)
- `gt3`: after `wait_until_completed()` (after line ~448)
- `gt4` + print: before return (before line ~462)

Use label `coalesced` instead of `linear_tg`.

**Step 3: Verify it compiles and runs**

Run:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal,timing --release \
  --test bench_degrees -- fast_d17 --nocapture --test-threads=1 2>&1 | grep "merkle-gpu"
```
Expected: Both `[merkle-gpu h=20 ...]` and `[merkle-gpu-inner h=20 linear_tg]` lines appear.

**Step 4: Commit**

```bash
git add plonky2/src/hash/metal/merkle.rs
git commit -m "feat(timing): add device-side GPU Merkle profiling"
```

---

### Task 3: Add timing to CPU Merkle path

**Files:**
- Modify: `plonky2/src/hash/merkle_tree.rs:127-260`

The CPU path runs `fill_subtree` per subtree in parallel via `par_chunks_exact_mut`. Inside each `fill_subtree`, there are two phases:
1. Leaf hashing (lines 152-161): parallel `hash_or_noop` for all leaves
2. Internal node hashing (lines 163-185): level-by-level `two_to_one`

Since subtrees run in parallel, we use `AtomicU64` accumulators to sum timing across all subtrees, then print the totals.

**Step 1: Add atomic timing to `fill_digests_buf`**

After the `digests_buf.is_empty()` early return (line 220), before the `subtree_digests_len` calculation (line 222), add timing setup:

```rust
#[cfg(feature = "timing")]
let cpu_t0 = std::time::Instant::now();
```

Replace the existing `for_each` closure (lines 228-239) with a version that tracks leaf vs internal timing using atomics:

```rust
#[cfg(feature = "timing")]
let leaf_ns = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "timing")]
let internal_ns = std::sync::atomic::AtomicU64::new(0);

digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
    |((subtree_digests, subtree_cap), subtree_leaves)| {
        subtree_cap.write(fill_subtree::<F, H>(
            subtree_digests,
            subtree_leaves,
            leaf_size,
            #[cfg(feature = "timing")]
            &leaf_ns,
            #[cfg(feature = "timing")]
            &internal_ns,
        ));
    },
);
```

Wait — this requires changing `fill_subtree`'s signature. A simpler approach: add timing INSIDE `fill_subtree` using thread-local or atomic accumulators passed via a global.

Actually, the simplest approach: modify `fill_subtree` to accept optional atomic refs.

**Revised approach — use a wrapper function:**

Add a `fill_subtree_timed` that wraps `fill_subtree` with timing:

```rust
#[cfg(feature = "timing")]
fn fill_subtree_timed<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[F],
    leaf_size: usize,
    leaf_nanos: &std::sync::atomic::AtomicU64,
    internal_nanos: &std::sync::atomic::AtomicU64,
) -> H::Hash {
    let leaves_count = leaves.len() / leaf_size;

    if leaves_count <= 2 {
        // Tiny subtrees — just delegate, no point timing
        return fill_subtree::<F, H>(digests_buf, leaves, leaf_size);
    }

    // Phase 1: leaf hashing
    let t0 = std::time::Instant::now();
    let (_, digests_leaves) = digests_buf.split_at_mut(digests_buf.len() - leaves_count);
    digests_leaves
        .into_par_iter()
        .enumerate()
        .for_each(|(leaf_idx, digest)| {
            let (_, r) = leaves.split_at(leaf_idx * leaf_size);
            let (leaf, _) = r.split_at(leaf_size);
            digest.write(H::hash_or_noop(leaf));
        });
    let t1 = std::time::Instant::now();

    // Phase 2: internal nodes
    let mut last_index = digests_buf.len() - leaves_count;
    for level_log in range(1, log2_strict(leaves_count)).rev() {
        let level_size = 1 << level_log;
        let (_, digests_slice) = digests_buf.split_at_mut(last_index - level_size);
        let (digests_slice, next_digests) = digests_slice.split_at_mut(level_size);
        digests_slice
            .into_par_iter()
            .zip(last_index - level_size..last_index)
            .for_each(|(digest, idx)| {
                let left_idx = 2 * (idx + 1) - last_index;
                let right_idx = left_idx + 1;
                unsafe {
                    let left_digest = next_digests[left_idx].assume_init();
                    let right_digest = next_digests[right_idx].assume_init();
                    digest.write(H::two_to_one(left_digest, right_digest));
                }
            });
        last_index -= level_size;
    }
    let t2 = std::time::Instant::now();

    leaf_nanos.fetch_add(
        t1.duration_since(t0).as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    internal_nanos.fetch_add(
        t2.duration_since(t1).as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );

    // Return cap hash
    unsafe {
        let left_digest = digests_buf[0].assume_init();
        let right_digest = digests_buf[1].assume_init();
        H::two_to_one(left_digest, right_digest)
    }
}
```

Then in `fill_digests_buf`, conditionally call `fill_subtree_timed` and print:

```rust
#[cfg(feature = "timing")]
let cpu_t0 = std::time::Instant::now();
#[cfg(feature = "timing")]
let leaf_nanos = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "timing")]
let internal_nanos = std::sync::atomic::AtomicU64::new(0);

digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
    |((subtree_digests, subtree_cap), subtree_leaves)| {
        #[cfg(feature = "timing")]
        {
            subtree_cap.write(fill_subtree_timed::<F, H>(
                subtree_digests,
                subtree_leaves,
                leaf_size,
                &leaf_nanos,
                &internal_nanos,
            ));
        }
        #[cfg(not(feature = "timing"))]
        {
            subtree_cap.write(fill_subtree::<F, H>(
                subtree_digests,
                subtree_leaves,
                leaf_size,
            ));
        }
    },
);

#[cfg(feature = "timing")]
{
    let cpu_total = cpu_t0.elapsed();
    let leaves_count = leaves.len() / leaf_size;
    let tree_height = log2_strict(leaves_count);
    let leaf_ms = leaf_nanos.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1_000_000.0;
    let internal_ms = internal_nanos.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1_000_000.0;
    eprintln!(
        "[merkle-cpu h={}] leaf_hash={:.1}ms internal_nodes={:.1}ms wall={:.1}ms subtrees={}",
        tree_height,
        leaf_ms,
        internal_ms,
        cpu_total.as_secs_f64() * 1000.0,
        1 << cap_height,
    );
}
```

Note: `leaf_ms` and `internal_ms` are **cumulative across all parallel subtrees** (not wall time). The `wall` value is the actual wall-clock time. The ratio of cumulative/wall shows parallelism efficiency.

**Step 2: Verify it compiles and runs**

Run with CPU-only + timing:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features timing --release \
  --test bench_degrees -- fast_d17 --nocapture --test-threads=1 2>&1 | grep "merkle-cpu"
```
Expected: Lines like `[merkle-cpu h=20] leaf_hash=X.Xms internal_nodes=X.Xms wall=X.Xms subtrees=16`

Run with Metal + timing (CPU path called for preprocessing tree):
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal,timing --release \
  --test bench_degrees -- fast_d17 --nocapture --test-threads=1 2>&1 | grep "merkle-"
```
Expected: Both `[merkle-gpu ...]` and `[merkle-cpu ...]` lines (CPU used for small preprocessing trees).

**Step 3: Commit**

```bash
git add plonky2/src/hash/merkle_tree.rs
git commit -m "feat(timing): add CPU Merkle leaf vs internal node profiling"
```

---

### Task 4: Run full profiling benchmarks and analyze

**Files:**
- Create: `docs/profiling-results-d13-d18.md`

**Step 1: Run Metal d13-d18 with timing**

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal,timing --release \
  --test bench_degrees -- full_d13 full_d14 full_d15 full_d16 full_d17 full_d18 \
  --nocapture --test-threads=1 2>&1 | tee /tmp/metal_timing.txt
```

**Step 2: Run CPU-only d13-d18 with timing**

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features timing --release \
  --test bench_degrees -- full_d13 full_d14 full_d15 full_d16 full_d17 full_d18 \
  --nocapture --test-threads=1 2>&1 | tee /tmp/cpu_timing.txt
```

**Step 3: Analyze and document**

Extract timing lines, compute percentages, identify bottlenecks. Write results to `docs/profiling-results-d13-d18.md`.

Key questions to answer:
- What fraction of GPU Merkle time is alloc vs encode vs execute vs readback?
- Is `wrap_or_copy` a significant cost?
- Is digest copy-back significant?
- In CPU path, what's the ratio of leaf hashing to internal node hashing?
- How does parallelism efficiency scale with tree height?

**Step 4: Commit**

```bash
git add docs/profiling-results-d13-d18.md
git commit -m "docs: add fine-grained Merkle profiling results for d13-d18"
```
