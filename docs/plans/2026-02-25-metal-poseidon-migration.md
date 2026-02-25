# Metal Poseidon Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate Metal-accelerated Poseidon/Merkle tree building from `plonky2-metal-demo` into `plonky2-fork/plonky2/` behind a `metal` feature flag, following the same pattern as the existing `cuda` feature.

**Architecture:** Copy `tracking`, `buffer_pool`, `threadgroup_config`, `utils`, `runtime`, and `merkle` modules from `plonky2-metal-demo/src/gpu/metal/` into `plonky2-fork/plonky2/src/hash/metal/`, adapting import paths. Copy the pre-compiled `poseidon_merkle_hasher_linear_threadgroup.metallib` into `plonky2/shaders/` (embedded via `include_bytes!`). Wire into `merkle_tree.rs` with a `fill_digests_buf_metal` function that falls back to CPU for wrong hasher type, small trees (< 2^13 leaves), and large trees (>= 2^21 leaves).

**Tech Stack:** Rust nightly-2024-01-16, `metal = "0.26.0"`, `once_cell` (already a dep), Apple Metal API, MSL shaders (pre-compiled to `.metallib`).

---

## Path Reference

- Source demo: `/Users/fujie/coding/cysic/20260220/plonky2-metal-demo/`
- Target fork: `/Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2/`
- Run all cargo commands from: `plonky2-fork/plonky2/` using toolchain `nightly-2024-01-16`

---

## Task 1: Add `metal` feature to `Cargo.toml`

**Files:**
- Modify: `plonky2-fork/plonky2/Cargo.toml`

**Step 1: Add feature and dependency**

In the `[features]` section, add:
```toml
metal = ["dep:metal"]
```

In the `[dependencies]` section, add:
```toml
metal = { version = "0.26.0", optional = true }
```

The `metal` crate only works on macOS (it links to the Metal framework). No additional `cfg(target_os)` guards are needed in Cargo.toml because the feature is opt-in.

**Step 2: Verify it compiles with the new feature**

```bash
cd /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2
cargo check --features metal 2>&1 | head -30
```

Expected: compiles (no errors yet; `metal` module doesn't exist yet so only unused dep warning is fine at this stage — or it may error on `mod metal` once we add it, but right now just the dep should be fine).

**Step 3: Commit**

```bash
git add plonky2/Cargo.toml
git commit -m "feat(metal): add metal feature flag and dependency"
```

---

## Task 2: Copy Metal shader files

**Files:**
- Create: `plonky2-fork/plonky2/shaders/` (new directory)

**Step 1: Copy required shader files**

```bash
mkdir -p /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2/shaders

# Pre-compiled shader library (embedded at build time)
cp /Users/fujie/coding/cysic/20260220/plonky2-metal-demo/shaders/poseidon_merkle_hasher_linear_threadgroup.metallib \
   /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2/shaders/

# Metal source files (for reference and recompilation)
cp /Users/fujie/coding/cysic/20260220/plonky2-metal-demo/shaders/poseidon_merkle_hasher_linear_threadgroup.metal \
   /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2/shaders/
cp /Users/fujie/coding/cysic/20260220/plonky2-metal-demo/shaders/poseidon_goldilocks.metal \
   /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2/shaders/
cp /Users/fujie/coding/cysic/20260220/plonky2-metal-demo/shaders/poseidon_goldilocks_mds.metal \
   /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2/shaders/
cp /Users/fujie/coding/cysic/20260220/plonky2-metal-demo/shaders/poseidon_goldilocks_threadgroup.metal \
   /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2/shaders/
cp /Users/fujie/coding/cysic/20260220/plonky2-metal-demo/shaders/goldilocks.metal \
   /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2/shaders/
cp /Users/fujie/coding/cysic/20260220/plonky2-metal-demo/shaders/u128.h.metal \
   /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2/shaders/
```

**Step 2: Verify the metallib exists**

```bash
ls -la /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2/shaders/
```

Expected: `poseidon_merkle_hasher_linear_threadgroup.metallib` present and non-zero size.

**Step 3: Commit**

```bash
git add plonky2/shaders/
git commit -m "feat(metal): add Metal shader files for linear+threadgroup Poseidon"
```

---

## Task 3: Write the failing integration test (TDD)

**Files:**
- Modify: `plonky2-fork/plonky2/src/hash/merkle_tree.rs`

**Step 1: Add the Metal correctness test to the `tests` module**

In `merkle_tree.rs`, find the `#[cfg(test)] mod tests { ... }` block (around line 842). Add this test at the end of the block, before the closing `}`:

```rust
#[cfg(feature = "metal")]
#[test]
fn test_merkle_trees_metal_poseidon_g64() -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    // 2^14 = 16384 leaves — above GPU threshold (2^13), hits Metal path
    let log_n = 14;
    let n = 1 << log_n;
    let leaves = random_data::<F>(n, 7);

    // verify_all_leaves builds the tree via new_from_2d → fill_digests_buf_meta
    // (which routes to Metal when feature = "metal") then verifies every proof.
    verify_all_leaves::<F, C, D>(leaves, 1)?;

    Ok(())
}
```

**Step 2: Run the test to confirm it fails (compile error — no `metal` module yet)**

```bash
cd /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2
cargo test --features metal test_merkle_trees_metal_poseidon_g64 2>&1 | tail -20
```

Expected: compile error, something like `unresolved import` or `module not found` — the test exists but the implementation doesn't yet.

**Step 3: Commit the test**

```bash
git add plonky2/src/hash/merkle_tree.rs
git commit -m "test(metal): add failing Metal Poseidon Merkle correctness test"
```

---

## Task 4: Create `src/hash/metal/tracking.rs`

**Files:**
- Create: `plonky2-fork/plonky2/src/hash/metal/tracking.rs`

**Step 1: Copy and adapt**

Copy `plonky2-metal-demo/src/gpu/metal/tracking.rs` to the new location. Make one change:

Remove the line:
```rust
use crate::gpu::metal::buffer_pool::BUFFER_POOL;
```

Replace with:
```rust
use crate::hash::metal::buffer_pool::BUFFER_POOL;
```

No other changes needed — `tracking.rs` has no other crate-internal imports.

**Step 2: Commit**

```bash
git add plonky2/src/hash/metal/tracking.rs
git commit -m "feat(metal): add tracking module (TrackedBuffer RAII)"
```

---

## Task 5: Create `src/hash/metal/threadgroup_config.rs`

**Files:**
- Create: `plonky2-fork/plonky2/src/hash/metal/threadgroup_config.rs`

**Step 1: Copy verbatim**

Copy `plonky2-metal-demo/src/gpu/metal/threadgroup_config.rs` to the new location without changes. This file has no crate-internal imports (only `std`).

**Step 2: Commit**

```bash
git add plonky2/src/hash/metal/threadgroup_config.rs
git commit -m "feat(metal): add threadgroup_config module"
```

---

## Task 6: Create `src/hash/metal/buffer_pool.rs`

**Files:**
- Create: `plonky2-fork/plonky2/src/hash/metal/buffer_pool.rs`

**Step 1: Copy and adapt**

Copy `plonky2-metal-demo/src/gpu/metal/buffer_pool.rs` to the new location.

Remove the `PersistentMerkleBuffers` struct and its `impl` block (it is only used by the coalesced path, which we are not porting). Also remove the `#[cfg(feature = "metal-bandwidth-opt")]` attribute at the top of that struct if present.

The remaining contents should be: `BufferPool` struct + `BUFFER_POOL` static + their `impl` blocks.

Change any import `use crate::gpu::metal::*` to `use crate::hash::metal::*` if present (check: `buffer_pool.rs` likely has no crate-internal imports).

**Step 2: Commit**

```bash
git add plonky2/src/hash/metal/buffer_pool.rs
git commit -m "feat(metal): add buffer_pool module (persistent GPU buffer reuse)"
```

---

## Task 7: Create `src/hash/metal/utils.rs`

**Files:**
- Create: `plonky2-fork/plonky2/src/hash/metal/utils.rs`

**Step 1: Create trimmed utils.rs**

This file is **not** copied verbatim — create it with only what `merkle.rs` will need. Write:

```rust
//! Utility types and functions for Metal GPU Merkle tree implementation.

use metal::MTLSize;
use plonky2_field::goldilocks_field::GoldilocksField;
use crate::hash::hash_types::HashOut;
use std::ptr;

/// Uniforms passed to linear+threadgroup Metal shaders. Must match `LinearUniforms` in the MSL shader.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LinearUniforms {
    pub level: u32,
    pub subtree_digests_len: u32,
    pub subtree_leaves_len: u32,
    pub leaf_size: u32,
    pub leaf_count: u32,
    pub subtree_count: u32,
    pub grid_width: u32,
}

/// Poseidon threadgroup memory constants (must match shader).
pub const POSEIDON_RC_SIZE: usize = 360 * std::mem::size_of::<u64>();   // 2880 bytes
pub const POSEIDON_MDS_SIZE: usize = 12 * std::mem::size_of::<i64>();   // 96 bytes
pub const POSEIDON_CONST_SIZE: usize = POSEIDON_RC_SIZE + POSEIDON_MDS_SIZE; // 2976 bytes

/// Copy raw data from a GPU buffer pointer into a new Vec.
///
/// # Safety
/// Caller must ensure `ptr` points to at least `elts` initialized `T` values.
pub unsafe fn from_buf_raw<T>(ptr: *const T, elts: usize) -> Vec<T> {
    let mut dst = Vec::with_capacity(elts);
    ptr::copy(ptr, dst.as_mut_ptr(), elts);
    dst.set_len(elts);
    dst
}

/// Compute a Metal dispatch grid large enough to cover `count` threads.
/// Returns a 1D or 2D MTLSize that fits within Metal's max threadgroup dimension.
pub fn get_size_for_count(count: usize) -> MTLSize {
    const MAX_DIM: usize = 32768;
    if count <= MAX_DIM {
        MTLSize { width: count as u64, height: 1, depth: 1 }
    } else {
        MTLSize {
            width: MAX_DIM as u64,
            height: ((count + MAX_DIM - 1) / MAX_DIM) as u64,
            depth: 1,
        }
    }
}

/// Return the index of a node in plonky2's recursive-layout digest buffer.
pub fn get_node_hash_index_in_digests(
    num_layers: usize,
    tree_length: usize,
    level: usize,
    index: usize,
) -> usize {
    let leaf_index = index << level;
    let tree_index = index >> (num_layers - level);
    let pair_index = (leaf_index & ((1 << num_layers) - 1)) >> level;
    let parity = pair_index & 1;
    let pair_index = pair_index >> 1;
    let siblings_index = (pair_index << (level + 1)) + (1 << level) - 1;
    let d_index = 2 * siblings_index + parity;
    d_index + (tree_index * tree_length)
}
```

**Step 2: Commit**

```bash
git add plonky2/src/hash/metal/utils.rs
git commit -m "feat(metal): add utils module (LinearUniforms, dispatch helpers)"
```

---

## Task 8: Create `src/hash/metal/runtime.rs`

**Files:**
- Create: `plonky2-fork/plonky2/src/hash/metal/runtime.rs`

**Step 1: Create simplified runtime (linear+threadgroup only)**

Copy `plonky2-metal-demo/src/gpu/metal/runtime.rs`. Then make the following edits:

1. **Replace `lazy_static` with `once_cell`**. Remove:
   ```rust
   use lazy_static::lazy_static;
   ```
   Add:
   ```rust
   use once_cell::sync::Lazy;
   ```

2. **Change import paths** (two lines near the top):
   ```rust
   // Before:
   use crate::gpu::metal::buffer_pool::{BufferPool, BUFFER_POOL, PersistentMerkleBuffers};
   use crate::gpu::metal::tracking::{track_allocation, track_deallocation, TrackedBuffer};
   // After:
   use crate::hash::metal::buffer_pool::{BufferPool, BUFFER_POOL};
   use crate::hash::metal::tracking::{track_allocation, track_deallocation, TrackedBuffer};
   ```

3. **Remove all non-linear-threadgroup shader constants**. Delete:
   - `const SHADERLIB: &[u8] = include_bytes!(...)` (base shader)
   - `#[cfg(feature = "metal-linear-merkle")] const SHADERLIB_LINEAR` block
   - `#[cfg(feature = "metal-threadgroup")] const SHADERLIB_THREADGROUP` block
   - `#[cfg(feature = "metal-bandwidth-opt")] const SHADERLIB_COALESCED` block

   Keep only (and remove the `#[cfg(feature = "metal-optimized")]` attribute gate):
   ```rust
   const SHADERLIB_LINEAR_THREADGROUP: &[u8] = include_bytes!(concat!(
       env!("CARGO_MANIFEST_DIR"),
       "/shaders/poseidon_merkle_hasher_linear_threadgroup.metallib"
   ));
   ```

4. **Strip the `MetalRuntime` struct** to only the linear+threadgroup fields. The final struct should be:
   ```rust
   pub struct MetalRuntime {
       pub device: Mutex<Device>,
       #[allow(dead_code)]
       f_poseidon_hash_leaves_linear_threadgroup: Function,
       #[allow(dead_code)]
       f_poseidon_hash_tree_level_linear_threadgroup: Function,
       #[allow(dead_code)]
       f_poseidon_hash_caps_linear_threadgroup: Function,
       pub(crate) pso_poseidon_hash_leaves_linear_threadgroup: ComputePipelineState,
       pub(crate) pso_poseidon_hash_tree_level_linear_threadgroup: ComputePipelineState,
       pub(crate) pso_poseidon_hash_caps_linear_threadgroup: ComputePipelineState,
       pub(crate) command_queue: CommandQueue,
       init_time: u128,
   }
   ```

5. **Replace `lazy_static! { pub static ref RUNTIME: ... }` with**:
   ```rust
   pub static RUNTIME: Lazy<MetalRuntime> = Lazy::new(|| {
       let device = Device::system_default().unwrap();
       let lib_lt = device.new_library_with_data(SHADERLIB_LINEAR_THREADGROUP).unwrap();
       let command_queue = device.new_command_queue();

       let f_leaves = lib_lt.get_function("poseidon_hash_leaves_linear_threadgroup", None).unwrap();
       let f_level  = lib_lt.get_function("poseidon_hash_tree_level_linear_threadgroup", None).unwrap();
       let f_caps   = lib_lt.get_function("poseidon_hash_caps_linear_threadgroup", None).unwrap();

       MetalRuntime {
           device: Mutex::new(Device::system_default().unwrap()),
           command_queue,
           pso_poseidon_hash_leaves_linear_threadgroup: create_pso(&device, &f_leaves),
           pso_poseidon_hash_tree_level_linear_threadgroup: create_pso(&device, &f_level),
           pso_poseidon_hash_caps_linear_threadgroup: create_pso(&device, &f_caps),
           f_poseidon_hash_leaves_linear_threadgroup: f_leaves,
           f_poseidon_hash_tree_level_linear_threadgroup: f_level,
           f_poseidon_hash_caps_linear_threadgroup: f_caps,
           init_time: std::time::SystemTime::now()
               .duration_since(std::time::UNIX_EPOCH)
               .unwrap()
               .as_micros(),
       }
   });
   ```

6. **Strip the `impl MetalRuntime`** to remove all `#[cfg(feature = "metal-*")]` getter methods except:
   - Keep all buffer allocation methods (`alloc`, `alloc_aligned`, `alloc_aligned_with_pool_hint`, `alloc_aligned_tracked`, `alloc_aligned_tracked_with_pool_hint`, `alloc_tracked`, `alloc_with_data`, `alloc_with_data_tracked`, `return_buffer`, `return_tracked_buffer`, `align_to_256`)
   - Keep `get()`, `warm_up()`
   - Remove: `get_persistent_buffers`, `persistent_buffers_fit`, `get_persistent_buffer_addresses` (coalesced only)
   - Keep (removing the `#[cfg(feature = "metal-optimized")]` gate):
     ```rust
     pub fn get_poseidon_hash_leaves_linear_threadgroup_pipeline_state(&self) -> &ComputePipelineState {
         &self.pso_poseidon_hash_leaves_linear_threadgroup
     }
     pub fn get_poseidon_hash_tree_level_linear_threadgroup_pipeline_state(&self) -> &ComputePipelineState {
         &self.pso_poseidon_hash_tree_level_linear_threadgroup
     }
     pub fn get_poseidon_hash_caps_linear_threadgroup_pipeline_state(&self) -> &ComputePipelineState {
         &self.pso_poseidon_hash_caps_linear_threadgroup
     }
     ```
   - Remove all other `#[cfg(feature = "metal-*")]` getter variants

**Step 2: Commit**

```bash
git add plonky2/src/hash/metal/runtime.rs
git commit -m "feat(metal): add Metal runtime (linear+threadgroup pipeline only)"
```

---

## Task 9: Create `src/hash/metal/merkle.rs`

**Files:**
- Create: `plonky2-fork/plonky2/src/hash/metal/merkle.rs`

**Step 1: Build from two source files**

`merkle.rs` combines the `convert_linear_to_plonky2_digests` helper (from `merkle_linear.rs`) with the `hash_merkle_tree_linear_threadgroup_buf_ho` dispatch function (from `merkle_linear_threadgroup.rs`).

**Imports at the top of the new file:**

```rust
//! GPU Merkle tree construction using linear+threadgroup Metal shader.

use metal::*;
use metal::objc::rc::autoreleasepool;
use plonky2_field::goldilocks_field::GoldilocksField;

use crate::hash::hash_types::HashOut;
use crate::hash::metal::runtime::MetalRuntime;
use crate::hash::metal::threadgroup_config::get_merkle_threadgroup_size;
use crate::hash::metal::tracking::track_deallocation;
use crate::hash::metal::utils::{
    from_buf_raw, get_node_hash_index_in_digests, get_size_for_count,
    LinearUniforms, POSEIDON_CONST_SIZE,
};
```

**Content — two `impl MetalRuntime` blocks:**

1. Copy `convert_linear_to_plonky2_digests` from `plonky2-metal-demo/src/gpu/metal/merkle_linear.rs` (lines 294–370), changing the plonky2 type imports as shown above (they now come from the local crate, not from `plonky2 = { path = "vendor/..." }`).

2. Copy `hash_merkle_tree_linear_threadgroup_buf_ho` from `plonky2-metal-demo/src/gpu/metal/merkle_linear_threadgroup.rs` (lines 100–305), making these changes:
   - Remove the outer `#![cfg(feature = "metal-optimized")]` attribute (file-level gate — delete it entirely)
   - Remove `new_merkle_tree_linear_threadgroup` (not needed; our entry point is `fill_digests_buf_metal` in `merkle_tree.rs`)
   - Remove `submit_merkle_tree_linear_threadgroup` and the `MerkleGpuJob` return variant (async path; not needed)
   - Remove `hash_leaves_linear_threadgroup_buf` (benchmarking helper; not needed)
   - Remove `reset_large_tree_counter` / `get_large_tree_count` (not needed)
   - Change `use crate::gpu::metal::*` imports to `use crate::hash::metal::*`
   - Change `plonky2::field::goldilocks_field::GoldilocksField` → `plonky2_field::goldilocks_field::GoldilocksField`
   - Change `plonky2::hash::hash_types::HashOut` → `crate::hash::hash_types::HashOut`

**Step 2: Commit**

```bash
git add plonky2/src/hash/metal/merkle.rs
git commit -m "feat(metal): add Merkle GPU dispatch (linear+threadgroup sync path)"
```

---

## Task 10: Create `src/hash/metal/mod.rs`

**Files:**
- Create: `plonky2-fork/plonky2/src/hash/metal/mod.rs`

**Step 1: Write mod.rs**

```rust
//! Metal GPU acceleration for Poseidon Merkle tree construction.
//!
//! Provides `fill_digests_buf_metal`, which is called from `merkle_tree.rs` when
//! the `metal` feature is enabled and CUDA is not.
//!
//! Only accelerates Poseidon/Goldilocks trees with 2^13–2^20 leaves.
//! All other cases fall through to the CPU implementation.

pub(crate) mod buffer_pool;
pub(crate) mod merkle;
pub(crate) mod runtime;
pub(crate) mod threadgroup_config;
pub(crate) mod tracking;
pub(crate) mod utils;

pub(crate) use runtime::RUNTIME;
```

Note: `fill_digests_buf_metal` is NOT defined here — it lives directly in `merkle_tree.rs` (same location as `fill_digests_buf_gpu`).

**Step 2: Commit**

```bash
git add plonky2/src/hash/metal/mod.rs
git commit -m "feat(metal): add metal module root (mod.rs)"
```

---

## Task 11: Wire `hash/mod.rs`

**Files:**
- Modify: `plonky2-fork/plonky2/src/hash/mod.rs`

**Step 1: Add the metal submodule declaration**

Open `hash/mod.rs`. After the existing `mod arch;` line (first line), add:

```rust
#[cfg(feature = "metal")]
pub(crate) mod metal;
```

**Step 2: Verify it compiles**

```bash
cd /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2
cargo check --features metal 2>&1 | head -40
```

Expected: the `metal` module compiles (may still have errors in `merkle_tree.rs` since `fill_digests_buf_metal` is not wired yet).

**Step 3: Commit**

```bash
git add plonky2/src/hash/mod.rs
git commit -m "feat(metal): declare metal module in hash/mod.rs"
```

---

## Task 12: Implement `fill_digests_buf_metal` and update dispatch in `merkle_tree.rs`

**Files:**
- Modify: `plonky2-fork/plonky2/src/hash/merkle_tree.rs`

### Step 1: Add Metal imports at the top of the file

Near the existing `#[cfg(feature = "cuda")]` imports block (around line 6), add:

```rust
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use metal::objc::rc::autoreleasepool;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use plonky2_field::goldilocks_field::GoldilocksField;
```

### Step 2: Update `fill_digests_buf_meta` — fix the CPU fallback cfg gate

Find the existing `#[cfg(not(feature = "cuda"))]` version of `fill_digests_buf_meta` (the one that simply calls `fill_digests_buf`). Change its attribute from:

```rust
#[cfg(not(feature = "cuda"))]
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
```

to:

```rust
#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
```

### Step 3: Add the Metal variant of `fill_digests_buf_meta`

Add this function immediately after the CPU variant (keep them adjacent):

```rust
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &Vec<F>,
    leaf_size: usize,
    cap_height: usize,
) {
    fill_digests_buf_metal::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
}
```

### Step 4: Add `fill_digests_buf_metal`

Add the following function near `fill_digests_buf_gpu` (or right after the `fill_digests_buf_meta` trio). This is the Metal dispatch implementation:

```rust
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn fill_digests_buf_metal<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &Vec<F>,
    leaf_size: usize,
    cap_height: usize,
) {
    use crate::hash::metal::{
        tracking::track_deallocation,
        RUNTIME,
    };
    use crate::plonk::config::HasherType;

    let leaf_count = leaves.len() / leaf_size;
    let tree_height = log2_strict(leaf_count);

    // GPU path is only valid for Poseidon/Goldilocks.
    // For all other hashers, fall through to CPU.
    if H::HASHER_TYPE != HasherType::Poseidon {
        fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
        return;
    }

    // All-cap trees have no internal digests; CPU handles them directly.
    // Small trees: Metal dispatch overhead exceeds compute benefit below 2^13 leaves.
    // Large trees: memory bandwidth saturation on M4 above 2^20 leaves.
    if cap_height == tree_height || tree_height < 13 || tree_height > 20 {
        fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
        return;
    }

    // Safety: when H::HASHER_TYPE == Poseidon, F = GoldilocksField (u64 newtype).
    // Both have identical representation, so the pointer cast is valid.
    let leaves_gl: &[GoldilocksField] = unsafe {
        std::slice::from_raw_parts(leaves.as_ptr() as *const GoldilocksField, leaves.len())
    };

    let (gpu_digests, gpu_caps) = autoreleasepool(|| {
        let leaves_buf = RUNTIME.alloc_with_data_tracked(leaves_gl);
        let result = RUNTIME.hash_merkle_tree_linear_threadgroup_buf_ho(
            leaves_buf.into_inner_untracked(),
            tree_height,
            leaf_size,
            cap_height,
        );
        track_deallocation(leaves_gl.len() * std::mem::size_of::<GoldilocksField>());
        result
    });

    // Safety: HashOut<GoldilocksField> and H::Hash have identical memory layout
    // when H::HASHER_TYPE == Poseidon (both are [u64; 4]).
    for (dst, src) in digests_buf.iter_mut().zip(gpu_digests.iter()) {
        dst.write(unsafe { *(src as *const _ as *const H::Hash) });
    }
    for (dst, src) in cap_buf.iter_mut().zip(gpu_caps.iter()) {
        dst.write(unsafe { *(src as *const _ as *const H::Hash) });
    }
}
```

### Step 5: Check it compiles

```bash
cd /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2
cargo check --features metal 2>&1 | head -40
```

Expected: clean compile (or only warnings).

### Step 6: Commit

```bash
git add plonky2/src/hash/merkle_tree.rs
git commit -m "feat(metal): wire Metal Poseidon dispatch into fill_digests_buf_meta"
```

---

## Task 13: Run tests and verify correctness

**Step 1: Run the Metal Poseidon test**

```bash
cd /Users/fujie/coding/cysic/20260220/plonky2-fork/plonky2
cargo test --features metal --release test_merkle_trees_metal_poseidon_g64 -- --nocapture 2>&1 | tail -30
```

Expected: `test hash::merkle_tree::tests::test_merkle_trees_metal_poseidon_g64 ... ok`

**Step 2: Run the full test suite without metal (CPU path must still pass)**

```bash
cargo test --release 2>&1 | tail -20
```

Expected: all pre-existing tests pass.

**Step 3: Run the full test suite with metal (Metal path + all existing tests)**

```bash
cargo test --features metal --release 2>&1 | tail -30
```

Expected: all tests pass including the new Metal test.

**Step 4: Commit if all tests pass**

```bash
git add -p  # review any last changes
git commit -m "feat(metal): Metal Poseidon Merkle acceleration — all tests passing

Migrated linear+threadgroup Metal GPU path from plonky2-metal-demo into
plonky2-fork. Gated by 'metal' feature flag, priority CUDA > Metal > CPU.
GPU path active for Poseidon/Goldilocks trees with 2^13–2^20 leaves.
Expected speedup: 1.3–1.45x vs CPU for typical proof sizes.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Troubleshooting

**Compile error: `SHADERLIB_LINEAR_THREADGROUP` not found at build time**
- Verify the `.metallib` file is at exactly `plonky2-fork/plonky2/shaders/poseidon_merkle_hasher_linear_threadgroup.metallib`
- The `include_bytes!` path is resolved relative to `CARGO_MANIFEST_DIR` (`plonky2-fork/plonky2/`)

**Link error: Metal framework not found**
- Only reproducible on macOS; the `metal` crate links to `Metal.framework` automatically on macOS

**Test panic: `cap height must be less than tree height`**
- The all-cap guard (`cap_height == tree_height`) in `fill_digests_buf_metal` must fire before calling the GPU path; verify the routing logic

**Wrong hashes (test failure)**
- The `convert_linear_to_plonky2_digests` layout conversion is the most likely source; verify the copy from `merkle_linear.rs` is exact
- Check that `get_node_hash_index_in_digests` in `utils.rs` is copied exactly from the demo

**Mutex poison in parallel tests**
- Metal runtime uses a `Mutex<Device>`; if a test panics while holding it, add `unwrap_or_else(|e| e.into_inner())` in `alloc`
