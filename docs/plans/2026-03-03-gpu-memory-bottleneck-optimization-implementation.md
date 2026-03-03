# GPU Memory Bottleneck Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address GPU memory bottlenecks (occupancy, unnecessary indirection, constant loading overhead) through 4 sequential optimizations, each independently validated by e2e benchmarks.

**Architecture:** Each optimization is tried in isolation. After implementation: compile shaders, run correctness tests, benchmark 3 runs Metal vs CPU. Keep if faster, revert if not.

**Tech Stack:** Metal Shading Language (MSL), Rust with `metal` crate, `xcrun` shader compiler.

---

## Baseline (pre-optimization, from 3-run benchmarks)

| Degree | CPU best (ms) | Metal best (ms) | Speedup |
|--------|--------------|-----------------|---------|
| d13 | 290 | 221 | 1.31x |
| d14 | 580 | 462 | 1.26x |
| d15 | 1,169 | 937 | 1.25x |
| d16 | 2,411 | 1,956 | 1.23x |
| d17 | 4,990 | 4,095 | 1.22x |
| d18 | 14,894 | 13,960 | 1.07x |

---

### Task 1: Revert Fast Partial Rounds

The fast partial rounds optimization added 5KB of threadgroup memory (2,976 → 8,088 bytes), reducing occupancy, with no measurable GPU speedup. Revert to clean baseline.

**Files:**
- Restore: `plonky2/shaders/goldilocks.metal` (from commit `5eae1ed8`)
- Restore: `plonky2/shaders/poseidon_goldilocks_threadgroup.metal` (from commit `5eae1ed8`)
- Restore: `plonky2/shaders/poseidon_merkle_hasher_linear_threadgroup.metal` (from commit `5eae1ed8`)
- Restore: `plonky2/shaders/poseidon_merkle_hasher_coalesced.metal` (from commit `5eae1ed8`)
- Restore: `plonky2/src/hash/metal/utils.rs` (from commit `5eae1ed8`)
- Delete: `plonky2/shaders/poseidon_fast_partial_constants.metal`

**Step 1: Restore pre-optimization shader and Rust files**

```bash
cd /Users/fujie/coding/cysic/20260220/plonky2-fork
git checkout 5eae1ed8 -- \
  plonky2/shaders/goldilocks.metal \
  plonky2/shaders/poseidon_goldilocks_threadgroup.metal \
  plonky2/shaders/poseidon_merkle_hasher_linear_threadgroup.metal \
  plonky2/shaders/poseidon_merkle_hasher_coalesced.metal \
  plonky2/src/hash/metal/utils.rs
```

**Step 2: Delete the fast partial constants header**

```bash
rm plonky2/shaders/poseidon_fast_partial_constants.metal
```

**Step 3: Recompile shaders**

```bash
cd plonky2/shaders

xcrun -sdk macosx metal -c poseidon_merkle_hasher_linear_threadgroup.metal \
  -o poseidon_merkle_hasher_linear_threadgroup.air
xcrun -sdk macosx metallib poseidon_merkle_hasher_linear_threadgroup.air \
  -o poseidon_merkle_hasher_linear_threadgroup.metallib
rm poseidon_merkle_hasher_linear_threadgroup.air

xcrun -sdk macosx metal -c poseidon_merkle_hasher_coalesced.metal \
  -o poseidon_merkle_hasher_coalesced.air
xcrun -sdk macosx metallib poseidon_merkle_hasher_coalesced.air \
  -o poseidon_merkle_hasher_coalesced.metallib
rm poseidon_merkle_hasher_coalesced.air

cd ../..
```

**Step 4: Run correctness tests**

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  -p plonky2 --lib -- test_merkle_trees_metal_poseidon_g64 --nocapture --test-threads=1

RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test metal_coalesced -- --nocapture --test-threads=1

RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test metal_proof -- --nocapture --test-threads=1
```

Expected: All pass (we're restoring known-good code).

**Step 5: Benchmark (3 alternating Metal/CPU runs + d18)**

```bash
# 3 runs of d13-d17 alternating Metal/CPU
for i in 1 2 3; do
  echo "--- Metal fast run $i ---"
  RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
    --test bench_degrees -- fast --nocapture --test-threads=1 2>&1 | grep "^degree"
  echo "--- CPU fast run $i ---"
  RUSTFLAGS="-Ctarget-cpu=native" cargo test --release \
    --test bench_degrees -- fast --nocapture --test-threads=1 2>&1 | grep "^degree"
done

# 3 runs of d18 alternating Metal/CPU
for i in 1 2 3; do
  echo "--- Metal d18 run $i ---"
  RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
    --test bench_degrees -- d18 --nocapture --test-threads=1 2>&1 | grep "^degree"
  echo "--- CPU d18 run $i ---"
  RUSTFLAGS="-Ctarget-cpu=native" cargo test --release \
    --test bench_degrees -- d18 --nocapture --test-threads=1 2>&1 | grep "^degree"
done
```

**Step 6: Compare with baseline**

Expected: Results should match the baseline table above (since this IS the baseline code). This confirms the revert is clean.

**Step 7: Commit**

```bash
git add -A
git commit -m "revert: remove fast partial rounds optimization (no GPU speedup, hurts occupancy)"
```

---

### Task 2: Remove shared_children Indirection in Linear Threadgroup Internal Kernel

The internal node kernel in `poseidon_merkle_hasher_linear_threadgroup.metal` writes 8 ulongs to threadgroup memory, hits a barrier, then reads back its own data. No cross-thread sharing occurs. Remove this indirection.

**Note:** The coalesced kernel (`poseidon_merkle_hasher_coalesced.metal`) uses a genuinely cooperative coalesced load where threads share data. Do NOT modify it.

**Files:**
- Modify: `plonky2/shaders/poseidon_merkle_hasher_linear_threadgroup.metal` (internal kernel only)
- Modify: `plonky2/src/hash/metal/merkle.rs:159-162` (remove child cache from TG memory)

**Step 1: Modify the internal kernel to read directly from device memory**

In `plonky2/shaders/poseidon_merkle_hasher_linear_threadgroup.metal`, replace lines 170-268 of the `poseidon_hash_tree_level_linear_threadgroup` kernel.

**Current code (lines 170-268):**
```metal
    // Split threadgroup memory: RC + MDS + fast partial constants, then child cache
    // Layout: [0..359] = round constants, [360..371] = MDS, [372..1010] = fast partial, [1011+] = child cache
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = (threadgroup long *)(tg_memory + POSEIDON_RC_TOTAL);
    threadgroup ulong * tg_fast_partial = tg_memory + POSEIDON_RC_TOTAL + MDS_CONST_TOTAL;
    threadgroup ulong * shared_children = tg_memory + POSEIDON_RC_TOTAL + MDS_CONST_TOTAL + FAST_PARTIAL_CONST_TOTAL;
    ...
    // Load children into threadgroup memory for better memory access patterns
    uint left_offset = left_child_idx * 4;
    uint right_offset = right_child_idx * 4;
    uint shared_base = local_id * 8;

    shared_children[shared_base + 0] = output[left_offset + 0];
    shared_children[shared_base + 1] = output[left_offset + 1];
    shared_children[shared_base + 2] = output[left_offset + 2];
    shared_children[shared_base + 3] = output[left_offset + 3];
    shared_children[shared_base + 4] = output[right_offset + 0];
    shared_children[shared_base + 5] = output[right_offset + 1];
    shared_children[shared_base + 6] = output[right_offset + 2];
    shared_children[shared_base + 7] = output[right_offset + 3];

    // Barrier to ensure all loads complete
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform hash using data from threadgroup memory and cached constants
    Fp p2_state[12];
    p2_state[0] = Fp(shared_children[shared_base + 0]);
    p2_state[1] = Fp(shared_children[shared_base + 1]);
    ...
```

**New code:** Remove the `shared_children` pointer, the write-to-TG, the barrier, and the read-from-TG. Read directly from device memory into `p2_state`:

After the child index calculation (line 233), replace everything up to the permutation call with:

```metal
    // Read children directly from device memory into registers
    uint left_offset = left_child_idx * 4;
    uint right_offset = right_child_idx * 4;

    Fp p2_state[12];
    p2_state[0] = Fp(output[left_offset + 0]);
    p2_state[1] = Fp(output[left_offset + 1]);
    p2_state[2] = Fp(output[left_offset + 2]);
    p2_state[3] = Fp(output[left_offset + 3]);
    p2_state[4] = Fp(output[right_offset + 0]);
    p2_state[5] = Fp(output[right_offset + 1]);
    p2_state[6] = Fp(output[right_offset + 2]);
    p2_state[7] = Fp(output[right_offset + 3]);
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;
```

Also remove `shared_children` from the threadgroup memory split at the top of the kernel (line 175). It's no longer needed.

**Step 2: Update Rust threadgroup memory size**

In `plonky2/src/hash/metal/merkle.rs`, find the internal level dispatch (around line 159-162):

**Current:**
```rust
            // Set threadgroup memory: round constants + MDS constants + child cache (8 ulongs per thread)
            let child_cache_size = (num_threads as usize) * 8 * std::mem::size_of::<u64>();
            let threadgroup_mem_size = POSEIDON_CONST_SIZE + child_cache_size;
            encoder.set_threadgroup_memory_length(0, threadgroup_mem_size as u64);
```

**New:**
```rust
            // Set threadgroup memory: round constants + MDS constants only (no child cache)
            encoder.set_threadgroup_memory_length(0, POSEIDON_CONST_SIZE as u64);
```

**Step 3: Update the header comment**

In `poseidon_merkle_hasher_linear_threadgroup.metal`, update lines 10-14:

**Current:**
```metal
// Threadgroup memory is used to cache:
// 1. Poseidon round constants (30 * 12 = 360 ulongs = 2880 bytes)
// 2. MDS frequency constants (12 longs = 96 bytes)
// 3. Child hash data for internal node computation (8 ulongs per thread)
// Total: ~3KB constants + child cache (well within 32KB Metal limit)
```

**New:**
```metal
// Threadgroup memory is used to cache:
// 1. Poseidon round constants (30 * 12 = 360 ulongs = 2880 bytes)
// 2. MDS frequency constants (12 longs = 96 bytes)
// Total: ~3KB constants (well within 32KB Metal limit)
```

**Step 4: Recompile shaders**

```bash
cd plonky2/shaders
xcrun -sdk macosx metal -c poseidon_merkle_hasher_linear_threadgroup.metal \
  -o poseidon_merkle_hasher_linear_threadgroup.air
xcrun -sdk macosx metallib poseidon_merkle_hasher_linear_threadgroup.air \
  -o poseidon_merkle_hasher_linear_threadgroup.metallib
rm poseidon_merkle_hasher_linear_threadgroup.air
cd ../..
```

(No need to recompile the coalesced shader — it wasn't modified.)

**Step 5: Run correctness tests**

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  -p plonky2 --lib -- test_merkle_trees_metal_poseidon_g64 --nocapture --test-threads=1

RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test metal_coalesced -- --nocapture --test-threads=1

RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test metal_proof -- --nocapture --test-threads=1
```

Expected: All pass. The hash output is identical — only the memory access path changed.

**Step 6: Benchmark (same protocol as Task 1 Step 5)**

Run 3 alternating Metal/CPU for d13-d17 fast + d18. Compare best `prove_min` with Task 1 baseline.

**Expected impact:** Internal kernel TG memory drops from ~11KB (2,976 + 8,192 child cache) to 2,976 bytes. Occupancy improvement for internal levels. May not be dramatic since internal levels are small relative to leaf hashing.

**Step 7: Decision**

If Metal times improved → commit:
```bash
git add plonky2/shaders/poseidon_merkle_hasher_linear_threadgroup.metal \
  plonky2/shaders/poseidon_merkle_hasher_linear_threadgroup.metallib \
  plonky2/src/hash/metal/merkle.rs
git commit -m "perf(metal): remove unnecessary shared_children indirection in linear TG internal kernel"
```

If no improvement → revert:
```bash
git checkout HEAD -- \
  plonky2/shaders/poseidon_merkle_hasher_linear_threadgroup.metal \
  plonky2/src/hash/metal/merkle.rs
# Recompile shader to restore
cd plonky2/shaders && xcrun -sdk macosx metal -c poseidon_merkle_hasher_linear_threadgroup.metal \
  -o poseidon_merkle_hasher_linear_threadgroup.air && \
  xcrun -sdk macosx metallib poseidon_merkle_hasher_linear_threadgroup.air \
  -o poseidon_merkle_hasher_linear_threadgroup.metallib && \
  rm poseidon_merkle_hasher_linear_threadgroup.air && cd ../..
```

---

### Task 3: Replace Threadgroup Constants with Direct Constant-Space Reads

Instead of cooperatively loading 2,976 bytes of constants into threadgroup memory at every kernel dispatch (with barriers), read directly from the compile-time `constant` arrays already embedded in the shader. Metal's constant address space is hardware-cached.

**Files:**
- Modify: `plonky2/shaders/poseidon_goldilocks_threadgroup.metal` (add new constant-read functions)
- Modify: `plonky2/shaders/poseidon_merkle_hasher_linear_threadgroup.metal` (all 3 kernels)
- Modify: `plonky2/shaders/poseidon_merkle_hasher_coalesced.metal` (all 4 kernels)
- Modify: `plonky2/src/hash/metal/merkle.rs` (remove TG memory for constants)
- Modify: `plonky2/src/hash/metal/utils.rs` (remove POSEIDON_CONST_SIZE or set to 0)

**Step 1: Add constant-read Poseidon functions**

In `plonky2/shaders/poseidon_goldilocks_threadgroup.metal`, add new functions that read from the compile-time `constant` arrays (`POSEIDON_RC_RAW` and `MDS_CONST_RAW`) instead of threadgroup pointers.

Add after the existing `poseidon_permute_tg_full` function (around line 391):

```metal
// ============================================================
// Constant-space Poseidon functions (no threadgroup memory)
// Read from compile-time constant arrays directly
// ============================================================

// Add round constants from constant-space array
inline void poseidon_add_rc_const(thread Fp* p2_state, int roundIndex) {
    uint base = roundIndex * POSEIDON_STATE_SIZE;
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        p2_state[i] = p2_state[i] + Fp(POSEIDON_RC_RAW[base + i]);
    }
}

// MDS block operations using constant-space MDS values
inline long3 block1_const(long3 in) {
    long3 b1 = long3(MDS_CONST_RAW[0], MDS_CONST_RAW[1], MDS_CONST_RAW[2]);
    return long3(
        in.x * b1.x + in.y * b1.z + in.z * b1.y,
        in.x * b1.y + in.y * b1.x + in.z * b1.z,
        in.x * b1.z + in.y * b1.y + in.z * b1.x
    );
}

inline tg_pair<long3, long3> block2_const(tg_pair<long3, long3> in) {
    long3 b2a = long3(MDS_CONST_RAW[3], MDS_CONST_RAW[4], MDS_CONST_RAW[5]);
    long3 b2b = long3(MDS_CONST_RAW[6], MDS_CONST_RAW[7], MDS_CONST_RAW[8]);

    long x0s = in.a.x + in.b.x;
    long x1s = in.a.y + in.b.y;
    long x2s = in.a.z + in.b.z;
    long y0s = b2a.x + b2b.x;
    long y1s = b2a.y + b2b.y;
    long y2s = b2a.z + b2b.z;

    long2 m0 = long2(in.a.x * b2a.x, in.b.x * b2b.x);
    long2 m1 = long2(in.a.y * b2a.z, in.b.y * b2b.z);
    long2 m2 = long2(in.a.z * b2a.y, in.b.z * b2b.y);
    long z0r = (m0.x - m0.y) + (x1s * y2s - m1.x - m1.y) + (x2s * y1s - m2.x - m2.y);
    long z0i = (x0s * y0s - m0.x - m0.y) + (-m1.x + m1.y) + (-m2.x + m2.y);

    m0 = long2(in.a.x * b2a.y, in.b.x * b2b.y);
    m1 = long2(in.a.y * b2a.x, in.b.y * b2b.x);
    m2 = long2(in.a.z * b2a.z, in.b.z * b2b.z);
    long z1r = (m0.x - m0.y) + (m1.x - m1.y) + (x2s * y2s - m2.x - m2.y);
    long z1i = (x0s * y1s - m0.x - m0.y) + (x1s * y0s - m1.x - m1.y) + (-m2.x + m2.y);

    m0 = long2(in.a.x * b2a.z, in.b.x * b2b.z);
    m1 = long2(in.a.y * b2a.y, in.b.y * b2b.y);
    m2 = long2(in.a.z * b2a.x, in.b.z * b2b.x);
    long z2r = (m0.x - m0.y) + (m1.x - m1.y) + (m2.x - m2.y);
    long z2i = (x0s * y2s - m0.x - m0.y) + (x1s * y1s - m1.x - m1.y) + (x2s * y0s - m2.x - m2.y);

    return { .a = long3(z0r, z1r, z2r), .b = long3(z0i, z1i, z2i) };
}

inline long3 block3_const(long3 in) {
    long3 b3 = long3(MDS_CONST_RAW[9], MDS_CONST_RAW[10], MDS_CONST_RAW[11]);
    return long3(
        in.x * b3.x - in.y * b3.z - in.z * b3.y,
        in.x * b3.y + in.y * b3.x - in.z * b3.z,
        in.x * b3.z + in.y * b3.y + in.z * b3.x
    );
}

inline void mds_multiply_freq_const(unsigned long state[12]) {
    long4 u0 = fft4_real_tg(ulong4(state[0], state[3], state[6], state[9]));
    long4 u1 = fft4_real_tg(ulong4(state[1], state[4], state[7], state[10]));
    long4 u2 = fft4_real_tg(ulong4(state[2], state[5], state[8], state[11]));

    long3 v0 = block1_const(long3(u0.x, u1.x, u2.x));
    tg_pair<long3, long3> v1 = block2_const({ .a = long3(u0.y, u1.y, u2.y), .b = long3(u0.z, u1.z, u2.z) });
    long3 v2 = block3_const(long3(u0.w, u1.w, u2.w));

    ulong4 s0 = ifft4_real_tg(long4(v0.x, v1.a.x, v1.b.x, v2.x));
    ulong4 s1 = ifft4_real_tg(long4(v0.y, v1.a.y, v1.b.y, v2.y));
    ulong4 s2 = ifft4_real_tg(long4(v0.z, v1.a.z, v1.b.z, v2.z));

    state[0] = s0.x; state[1] = s1.x; state[2] = s2.x;
    state[3] = s0.y; state[4] = s1.y; state[5] = s2.y;
    state[6] = s0.z; state[7] = s1.z; state[8] = s2.z;
    state[9] = s0.w; state[10] = s1.w; state[11] = s2.w;
}

inline void apply_mds_freq_const(thread Fp* shared, unsigned local_state_offset) {
    unsigned long state_l[12];
    unsigned long state_h[12];

    #pragma unroll
    for (unsigned j = 0; j < 12; j++) {
        Fp element = shared[local_state_offset + j];
        unsigned long s = (unsigned long)element;
        state_l[j] = s & 0xFFFFFFFF;
        state_h[j] = s >> 32;
    }

    mds_multiply_freq_const(state_l);
    mds_multiply_freq_const(state_h);

    u128 s = u128(state_l[0]) + (u128(state_h[0]) << 32);
    s.accumulate_mul_2_ulong(static_cast<ulong>(shared[0]), 8);
    ulong reduced = reduce128(s.high, s.low);
    shared[local_state_offset] = Fp(reduced < GOLDILOCKS_PRIME ? reduced : (reduced - GOLDILOCKS_PRIME));

    #pragma unroll
    for (unsigned j = 1; j < 12; j++) {
        s = u128(state_l[j]) + (u128(state_h[j]) << 32);
        reduced = reduce128(s.high, s.low);
        shared[local_state_offset + j] = Fp(reduced < GOLDILOCKS_PRIME ? reduced : (reduced - GOLDILOCKS_PRIME));
    }
}

inline void poseidon_full_round_const(thread Fp* p2_state, int roundIndex) {
    poseidon_add_rc_const(p2_state, roundIndex);
    poseidon_sbox_all_tg(p2_state);
    apply_mds_freq_const(p2_state, 0);
}

inline void poseidon_partial_round_const(thread Fp* p2_state, int roundIndex) {
    poseidon_add_rc_const(p2_state, roundIndex);
    p2_state[0] = p2_state[0].pow7();
    apply_mds_freq_const(p2_state, 0);
}

// Poseidon permutation reading all constants from constant address space
// No threadgroup memory required
inline void poseidon_permute_const(thread Fp* p2_state) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        poseidon_full_round_const(p2_state, i);
    }

    #pragma unroll
    for (int i = 4; i < 26; i++) {
        poseidon_partial_round_const(p2_state, i);
    }

    #pragma unroll
    for (int i = 26; i < 30; i++) {
        poseidon_full_round_const(p2_state, i);
    }
}
```

**Step 2: Update linear_threadgroup kernels**

In `plonky2/shaders/poseidon_merkle_hasher_linear_threadgroup.metal`:

**Leaf kernel** — Remove threadgroup memory parameter, constant loading, and barrier. Change permutation call:

Remove from kernel signature: `threadgroup ulong * tg_memory[[threadgroup(0)]]`

Remove all constant loading and TG memory splitting (lines 73-83). Remove the `tg_round_constants`, `tg_mds_constants` variables.

Replace `poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants)` with `poseidon_permute_const(p2_state)`.

**Internal kernel** — Same: remove TG memory for constants. If shared_children was already removed in Task 2, remove `threadgroup ulong * tg_memory[[threadgroup(0)]]` entirely. Replace permutation call with `poseidon_permute_const(p2_state)`.

**Caps kernel** — Same pattern: remove TG memory, use `poseidon_permute_const`.

**Step 3: Update coalesced kernels**

In `plonky2/shaders/poseidon_merkle_hasher_coalesced.metal`:

**Leaf kernel** — Remove TG memory for constants. Use `poseidon_permute_const`.

**Internal kernel** — Keep `threadgroup ulong * tg_memory[[threadgroup(0)]]` but only for `shared_children` (the cooperative coalesced load). Remove constant loading and barrier for constants. The TG memory layout changes from `[constants + child_cache]` to just `[child_cache]`.

Change the TG memory split from:
```metal
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = ...;
    threadgroup ulong * shared_children = tg_memory + POSEIDON_RC_TOTAL + MDS_CONST_TOTAL;
```
to:
```metal
    threadgroup ulong * shared_children = tg_memory;
```

Remove `load_all_constants_tg(...)` and the barrier for constants.

Replace `poseidon_permute_tg_full(...)` with `poseidon_permute_const(p2_state)`.

**Instrumented kernel** — Same changes as internal kernel.

**Caps kernel** — Remove TG memory entirely. Use `poseidon_permute_const`.

**Step 4: Update Rust dispatch code**

In `plonky2/src/hash/metal/merkle.rs`:

**Linear_threadgroup path (`hash_merkle_tree_linear_threadgroup_buf_ho`):**

Leaf kernel (line 121): Change from `POSEIDON_CONST_SIZE` to `0`:
```rust
encoder.set_threadgroup_memory_length(0, 0);
```

Internal kernel (lines 159-162): If Task 2 removed child cache, set to `0`. Otherwise just constants size:
```rust
// No threadgroup memory needed (constants read from constant space, no child cache)
encoder.set_threadgroup_memory_length(0, 0);
```

Caps kernel (line 207): Change to `0`:
```rust
encoder.set_threadgroup_memory_length(0, 0);
```

**Coalesced path (`hash_merkle_tree_coalesced_buf_ho`):**

Leaf kernel (line 358): Change to `0`:
```rust
encoder.set_threadgroup_memory_length(0, 0);
```

Internal kernel (lines 412-415): Change from `POSEIDON_CONST_SIZE + child_cache_size` to just `child_cache_size`:
```rust
let child_cache_size = (threadgroup_size as usize) * 8 * std::mem::size_of::<u64>();
encoder.set_threadgroup_memory_length(0, child_cache_size as u64);
```

Caps kernel (line 462): Change to `0`:
```rust
encoder.set_threadgroup_memory_length(0, 0);
```

**Step 5: Clean up utils.rs**

In `plonky2/src/hash/metal/utils.rs`, the `POSEIDON_CONST_SIZE` constant is no longer used. Either remove it or keep it with a comment. Check if `POSEIDON_RC_SIZE` and `POSEIDON_MDS_SIZE` are still referenced. Remove unused constants.

**Step 6: Recompile both shaders**

```bash
cd plonky2/shaders

xcrun -sdk macosx metal -c poseidon_merkle_hasher_linear_threadgroup.metal \
  -o poseidon_merkle_hasher_linear_threadgroup.air
xcrun -sdk macosx metallib poseidon_merkle_hasher_linear_threadgroup.air \
  -o poseidon_merkle_hasher_linear_threadgroup.metallib
rm poseidon_merkle_hasher_linear_threadgroup.air

xcrun -sdk macosx metal -c poseidon_merkle_hasher_coalesced.metal \
  -o poseidon_merkle_hasher_coalesced.air
xcrun -sdk macosx metallib poseidon_merkle_hasher_coalesced.air \
  -o poseidon_merkle_hasher_coalesced.metallib
rm poseidon_merkle_hasher_coalesced.air

cd ../..
```

**Step 7: Run correctness tests**

Same 3 test commands as Task 1 Step 4.

**Step 8: Benchmark**

Same benchmark protocol as Task 1 Step 5.

**Expected impact:** Eliminates all threadgroup memory for constants. Leaf/caps kernels use 0 bytes TG. Coalesced internal kernel uses only child_cache (~8KB). Maximum GPU occupancy.

**Step 9: Decision — keep or revert**

If faster → commit:
```bash
git add plonky2/shaders/ plonky2/src/hash/metal/merkle.rs plonky2/src/hash/metal/utils.rs
git commit -m "perf(metal): read Poseidon constants from constant space, eliminate threadgroup memory"
```

If not → revert:
```bash
git checkout HEAD -- plonky2/shaders/ plonky2/src/hash/metal/merkle.rs plonky2/src/hash/metal/utils.rs
# Recompile both shaders to restore
```

---

### Task 4: (Speculative) Reduce MDS Register Pressure

**Only attempt if Tasks 1-3 do not provide sufficient improvement.**

The `apply_mds_freq_const` function holds `state_l[12]` + `state_h[12]` + FFT intermediates simultaneously (~130+ 32-bit register equivalents). Apple Silicon GPUs have ~32 registers per thread, causing heavy register spilling to device memory.

**Approach:** Restructure to reuse a single `state_x[12]` array for both low and high halves, storing intermediate results back into `p2_state` temporarily.

This is architecturally complex and may not help if the constant cache improvements from Task 3 already resolve the memory pressure. Defer detailed implementation until Tasks 1-3 results are known.

---

## Verification Checklist

After each task:
- [ ] Shaders compile without warnings
- [ ] `test_merkle_trees_metal_poseidon_g64` passes (Merkle correctness heights 1-14)
- [ ] `metal_coalesced` passes 4/4 (heights 20-22)
- [ ] `metal_proof` passes 1/1 (e2e proof roundtrip)
- [ ] 3-run benchmark shows improvement (or decision to revert)
- [ ] Commit with descriptive message (or clean revert)
