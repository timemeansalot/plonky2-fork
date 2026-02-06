//! Linear layout + threadgroup optimized Merkle tree GPU implementation.
//!
//! Combines linear memory layout with threadgroup memory caching for optimal performance.

#![cfg(feature = "metal-optimized")]

use std::sync::atomic::{AtomicUsize, Ordering};
use metal::*;

/// Counter for large trees processed - used for hybrid GPU/CPU mode
/// At degree 25, only the first large tree uses GPU to reduce memory contention
static LARGE_TREE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Reset the large tree counter (call at start of proof generation)
pub fn reset_large_tree_counter() {
    LARGE_TREE_COUNT.store(0, Ordering::SeqCst);
}

/// Get current large tree count
pub fn get_large_tree_count() -> usize {
    LARGE_TREE_COUNT.load(Ordering::SeqCst)
}
use crate::{
    field::goldilocks_field::GoldilocksField,
    hash::{
        hash_types::HashOut,
        merkle_tree::{MerkleCap, MerkleTree},
        poseidon::PoseidonHash,
    },
};
use metal::objc::rc::autoreleasepool;

use crate::gpu::metal::merkle_job::MerkleGpuJob;
use crate::gpu::metal::runtime::MetalRuntime;
use crate::gpu::metal::threadgroup_config::get_merkle_threadgroup_size;
use crate::gpu::metal::tracking::track_deallocation;
use crate::gpu::metal::utils::{from_buf_raw, get_size_for_count, LinearUniforms, POSEIDON_CONST_SIZE};

const TG_CONST_FLAG: u32 = 1;
const TG_CHILD_CACHE_FLAG: u32 = 2;

fn tg_flags_from_env() -> u32 {
    let use_tg_consts = std::env::var("PLONKY2_TG_CONST")
        .map(|v| v != "0")
        .unwrap_or(true);
    let use_child_cache = std::env::var("PLONKY2_TG_CHILD_CACHE")
        .map(|v| v != "0")
        .unwrap_or(false);
    (if use_tg_consts { TG_CONST_FLAG } else { 0 })
        | (if use_child_cache { TG_CHILD_CACHE_FLAG } else { 0 })
}

impl MetalRuntime {
    /// Build a Merkle tree using linear layout + threadgroup memory optimization
    /// This combines the benefits of linear memory coalescing with threadgroup caching
    pub fn new_merkle_tree_linear_threadgroup(
        leaves: Vec<Vec<GoldilocksField>>,
        cap_height: usize,
    ) -> MerkleTree<GoldilocksField, PoseidonHash> {
        let leaf_length = leaves[0].len();
        let leaf_count = leaves.len();

        // Validate power-of-two leaf count (required for GPU implementation)
        assert!(
            leaf_count.is_power_of_two(),
            "GPU Merkle tree requires power-of-two leaf count, got {}",
            leaf_count
        );

        let tree_height = leaf_count.trailing_zeros() as usize;

        if std::env::var("PLONKY2_METAL_LOG").is_ok() {
            println!(
                "[GPU Merkle Linear+Threadgroup] Building tree: height={}, leaves={}, leaf_length={}, cap_height={}",
                tree_height, leaf_count, leaf_length, cap_height
            );
        }

        // Fall back to CPU for small trees or when cap_height equals tree_height
        if cap_height == tree_height || tree_height < 13 {
            if std::env::var("PLONKY2_METAL_LOG").is_ok() {
                println!("[GPU Merkle Linear+Threadgroup] Using CPU fallback");
            }
            return MerkleTree::new_from_2d(leaves, cap_height);
        }

        // At degree 25 (tree_height >= 21), CPU is more efficient for ALL large trees
        // due to better memory prefetching and cache coherency. GPU's advantage is
        // negated by memory bandwidth saturation at this scale.
        if tree_height >= 21 {
            if std::env::var("PLONKY2_METAL_LOG").is_ok() {
                println!(
                    "[GPU Merkle Linear+Threadgroup] height={}, k={} -> CPU (optimized for large trees)",
                    tree_height, leaf_length
                );
            }
            return MerkleTree::new_from_2d(leaves, cap_height);
        }

        if std::env::var("PLONKY2_METAL_LOG").is_ok() {
            println!("[GPU Merkle Linear+Threadgroup] Using GPU with linear layout + threadgroup optimization");
        }

        let leaf_u64s = leaves.concat();
        let (digests, caps) = autoreleasepool::<(Vec<HashOut<GoldilocksField>>, Vec<HashOut<GoldilocksField>>), _>(|| {
            // Use TrackedBuffer RAII for automatic tracking
            // Keep tracked until GPU work completes (hash function returns)
            let leaves_buffer = Self::get().alloc_with_data_tracked(&leaf_u64s);
            let result = Self::get().hash_merkle_tree_linear_threadgroup_buf_ho(
                leaves_buffer.into_inner_untracked(),
                tree_height,
                leaf_length,
                cap_height,
            );
            // Now GPU work is complete, track deallocation
            track_deallocation(leaf_u64s.len() * std::mem::size_of::<GoldilocksField>());
            result
        });

        MerkleTree::new_from_fields(leaf_u64s, leaf_length, digests, MerkleCap(caps))
    }

    /// Hash merkle tree with linear layout + threadgroup optimization
    pub fn hash_merkle_tree_linear_threadgroup_buf_ho(
        &self,
        leaves_buffer: Buffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
    ) -> (Vec<HashOut<GoldilocksField>>, Vec<HashOut<GoldilocksField>>) {
        let leaf_count = 1usize << tree_height;

        assert!(
            cap_height < tree_height,
            "cap height must be less than tree height"
        );

        let num_caps = 1usize << cap_height;
        let subtree_leaves_len = leaf_count >> cap_height;
        let subtree_digests_len = 2 * subtree_leaves_len - 1;
        let total_digests = subtree_digests_len * num_caps;
        let num_layers = tree_height - cap_height;

        // Use TrackedBuffer RAII for digests - auto-tracks allocation/deallocation
        let buffer_size = (total_digests * 4) * std::mem::size_of::<u64>();
        let (digests_buffer, digests_from_pool) = self.alloc_aligned_tracked_with_pool_hint(buffer_size);

        // Only zero new buffers; pooled buffers were written by GPU previously
        if !digests_from_pool {
            unsafe {
                let ptr = digests_buffer.contents() as *mut u8;
                std::ptr::write_bytes(ptr, 0, Self::align_to_256(buffer_size));
            }
        }

        // Use TrackedBuffer RAII for caps - auto-tracks allocation/deallocation
        let caps_size = (num_caps * 4) * std::mem::size_of::<u64>();
        let (caps_buffer, caps_from_pool) = self.alloc_aligned_tracked_with_pool_hint(caps_size);
        if !caps_from_pool {
            unsafe {
                let ptr = caps_buffer.contents() as *mut u8;
                std::ptr::write_bytes(ptr, 0, Self::align_to_256(caps_size));
            }
        }

        // Get pipeline states for linear+threadgroup kernels
        let use_fused_leaf_level1 = num_layers > 1;
        let pipeline_hash_leaves = if use_fused_leaf_level1 {
            self.get_poseidon_hash_leaves_and_level1_linear_threadgroup_pipeline_state()
        } else {
            self.get_poseidon_hash_leaves_linear_threadgroup_pipeline_state()
        };
        let pipeline_hash_tree_level = self.get_poseidon_hash_tree_level_linear_threadgroup_pipeline_state();
        let pipeline_hash_caps = self.get_poseidon_hash_caps_linear_threadgroup_pipeline_state();

        let command_buffer = self.command_queue.new_command_buffer();

        // Create uniforms (same struct as linear layout)
        let mut uniforms = LinearUniforms {
            level: 0,
            subtree_digests_len: subtree_digests_len as u32,
            subtree_leaves_len: subtree_leaves_len as u32,
            leaf_size: leaf_length as u32,
            leaf_count: leaf_count as u32,
            subtree_count: num_caps as u32,
            grid_width: 0,
        };

        let tg_flags = tg_flags_from_env();

        // Hash all leaves
        {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_leaves);
            encoder.set_buffer(0, Some(&leaves_buffer), 0);
            encoder.set_buffer(1, Some(&*digests_buffer), 0);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &tg_flags as *const u32 as *const core::ffi::c_void);

            let total_threads = if use_fused_leaf_level1 {
                leaf_count / 2
            } else {
                leaf_count
            };
            // Use tuned threadgroup size based on tree size, clamped to device limits
            let simd_width = pipeline_hash_leaves.thread_execution_width() as usize;
            let max_threads = pipeline_hash_leaves.max_total_threads_per_threadgroup() as usize;
            let num_threads = get_merkle_threadgroup_size(total_threads, simd_width, max_threads) as u64;
            let lg = (total_threads as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = get_size_for_count(lg as usize);

            uniforms.grid_width = (thread_group_count.width * num_threads) as u32;

            // Use TrackedBuffer RAII for uniforms - auto-tracks on drop
            let uniforms_buffer = self.alloc_with_data_tracked(&[uniforms]);
            encoder.set_buffer(2, Some(&*uniforms_buffer), 0);

            let thread_group_size = MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            };

            let tg_mem = if (tg_flags & TG_CONST_FLAG) != 0 {
                POSEIDON_CONST_SIZE
            } else {
                0
            };
            encoder.set_threadgroup_memory_length(0, tg_mem as u64);

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            // uniforms_buffer RAII: auto-tracks deallocation on drop
        }

        // Hash internal tree levels with threadgroup memory
        let start_level = if use_fused_leaf_level1 { 2 } else { 1 };
        for level in start_level..num_layers {
            uniforms.level = level as u32;

            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_tree_level);
            encoder.set_buffer(0, Some(&*digests_buffer), 0);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &tg_flags as *const u32 as *const core::ffi::c_void);

            let nodes_at_this_level = subtree_leaves_len >> level;
            let total_nodes = nodes_at_this_level * num_caps;
            // Use tuned threadgroup size, clamped to device limits
            let simd_width = pipeline_hash_tree_level.thread_execution_width() as usize;
            let max_threads = pipeline_hash_tree_level.max_total_threads_per_threadgroup() as usize;
            let num_threads = get_merkle_threadgroup_size(leaf_count, simd_width, max_threads) as u64;
            let lg = (total_nodes as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = get_size_for_count(lg as usize);

            uniforms.grid_width = (thread_group_count.width * num_threads) as u32;

            // Pass uniforms as bytes
            encoder.set_bytes(
                1,
                std::mem::size_of::<LinearUniforms>() as u64,
                &uniforms as *const LinearUniforms as *const core::ffi::c_void,
            );

            let mut tg_mem = 0usize;
            if (tg_flags & TG_CONST_FLAG) != 0 {
                tg_mem += POSEIDON_CONST_SIZE;
            }
            if (tg_flags & TG_CHILD_CACHE_FLAG) != 0 {
                tg_mem += (num_threads as usize) * 8 * std::mem::size_of::<u64>();
            }
            encoder.set_threadgroup_memory_length(0, tg_mem as u64);

            let thread_group_size = MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
        }

        // Compute cap hashes
        {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_caps);
            encoder.set_buffer(0, Some(&*caps_buffer), 0);
            encoder.set_buffer(1, Some(&*digests_buffer), 0);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &tg_flags as *const u32 as *const core::ffi::c_void);

            // Use TrackedBuffer RAII for uniforms - auto-tracks on drop
            let uniforms_buffer = self.alloc_with_data_tracked(&[uniforms]);
            encoder.set_buffer(2, Some(&*uniforms_buffer), 0);

            // Use tuned threadgroup size, clamped to device limits
            let simd_width = pipeline_hash_caps.thread_execution_width() as usize;
            let max_threads = pipeline_hash_caps.max_total_threads_per_threadgroup() as usize;
            let num_threads = get_merkle_threadgroup_size(num_caps, simd_width, max_threads) as u64;
            let lg = (num_caps as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = MTLSize {
                width: lg.max(1),
                height: 1,
                depth: 1,
            };
            let thread_group_size = MTLSize {
                width: num_threads.min(num_caps as u64),
                height: 1,
                depth: 1,
            };

            let tg_mem = if (tg_flags & TG_CONST_FLAG) != 0 {
                POSEIDON_CONST_SIZE
            } else {
                0
            };
            encoder.set_threadgroup_memory_length(0, tg_mem as u64);

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            // uniforms_buffer RAII: auto-tracks deallocation on drop
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Convert linear layout digests to plonky2's expected format
        let digests = self.convert_linear_to_plonky2_digests(
            &*digests_buffer,
            tree_height,
            cap_height,
            subtree_digests_len,
            subtree_leaves_len,
        );

        // Read caps
        let caps_ptr = (*caps_buffer).contents() as *mut HashOut<GoldilocksField>;
        let caps = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(caps_ptr, num_caps) };

        // Return TrackedBuffers to pool for reuse (tracks deallocation since buffer is no longer "in use")
        self.return_tracked_buffer(digests_buffer);
        self.return_tracked_buffer(caps_buffer);

        (digests, caps)
    }

    /// Submit async Merkle tree build using linear+threadgroup optimization
    /// Returns immediately with a job handle; call finish() to wait and get results
    /// This allows doing CPU work while GPU builds the tree
    pub fn submit_merkle_tree_linear_threadgroup(
        &self,
        leaves_buffer: Buffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
    ) -> MerkleGpuJob {
        let leaf_count = 1usize << tree_height;

        assert!(
            cap_height < tree_height,
            "cap height must be less than tree height"
        );

        let num_caps = 1usize << cap_height;
        let subtree_leaves_len = leaf_count >> cap_height;
        let subtree_digests_len = 2 * subtree_leaves_len - 1;
        let total_digests = subtree_digests_len * num_caps;

        // Allocate digests buffer with 256-byte alignment
        let buffer_size = (total_digests * 4) * std::mem::size_of::<u64>();
        let digests_buffer = self.alloc_aligned(buffer_size);
        unsafe {
            let ptr = digests_buffer.contents() as *mut u8;
            std::ptr::write_bytes(ptr, 0, Self::align_to_256(buffer_size));
        }

        // Allocate caps buffer with 256-byte alignment
        let caps_size = (num_caps * 4) * std::mem::size_of::<u64>();
        let caps_buffer = self.alloc_aligned(caps_size);
        unsafe {
            let ptr = caps_buffer.contents() as *mut u8;
            std::ptr::write_bytes(ptr, 0, Self::align_to_256(caps_size));
        }

        let num_layers = tree_height - cap_height;
        // Get pipeline states
        let use_fused_leaf_level1 = num_layers > 1;
        let pipeline_hash_leaves = if use_fused_leaf_level1 {
            self.get_poseidon_hash_leaves_and_level1_linear_threadgroup_pipeline_state()
        } else {
            self.get_poseidon_hash_leaves_linear_threadgroup_pipeline_state()
        };
        let pipeline_hash_tree_level = self.get_poseidon_hash_tree_level_linear_threadgroup_pipeline_state();
        let pipeline_hash_caps = self.get_poseidon_hash_caps_linear_threadgroup_pipeline_state();

        let command_buffer = self.command_queue.new_command_buffer();

        let mut uniforms = LinearUniforms {
            level: 0,
            subtree_digests_len: subtree_digests_len as u32,
            subtree_leaves_len: subtree_leaves_len as u32,
            leaf_size: leaf_length as u32,
            leaf_count: leaf_count as u32,
            subtree_count: num_caps as u32,
            grid_width: 0,
        };

        let tg_flags = tg_flags_from_env();

        // Hash all leaves
        {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_leaves);
            encoder.set_buffer(0, Some(&leaves_buffer), 0);
            encoder.set_buffer(1, Some(&digests_buffer), 0);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &tg_flags as *const u32 as *const core::ffi::c_void);

            let total_threads = if use_fused_leaf_level1 {
                leaf_count / 2
            } else {
                leaf_count
            };
            let simd_width = pipeline_hash_leaves.thread_execution_width() as usize;
            let max_threads = pipeline_hash_leaves.max_total_threads_per_threadgroup() as usize;
            let num_threads = get_merkle_threadgroup_size(total_threads, simd_width, max_threads) as u64;
            let lg = (total_threads as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = get_size_for_count(lg as usize);

            uniforms.grid_width = (thread_group_count.width * num_threads) as u32;
            encoder.set_bytes(2, std::mem::size_of::<LinearUniforms>() as u64,
                unsafe { std::mem::transmute(&uniforms as *const LinearUniforms) });

            let tg_mem = if (tg_flags & TG_CONST_FLAG) != 0 {
                POSEIDON_CONST_SIZE
            } else {
                0
            };
            encoder.set_threadgroup_memory_length(0, tg_mem as u64);

            let thread_group_size = MTLSize { width: num_threads, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
        }

        // Build tree levels
        let start_level = if use_fused_leaf_level1 { 2 } else { 1 };
        for level in start_level..num_layers {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_tree_level);

            uniforms.level = level as u32;

            let simd_width = pipeline_hash_tree_level.thread_execution_width() as usize;
            let max_threads = pipeline_hash_tree_level.max_total_threads_per_threadgroup() as usize;
            let num_threads = get_merkle_threadgroup_size(leaf_count, simd_width, max_threads) as u64;
            let nodes_at_level = (leaf_count >> level) as NSUInteger;
            let lg = (nodes_at_level + num_threads - 1) / num_threads;
            let thread_group_count = get_size_for_count(lg as usize);

            uniforms.grid_width = (thread_group_count.width * num_threads) as u32;
            encoder.set_bytes(1, std::mem::size_of::<LinearUniforms>() as u64,
                unsafe { std::mem::transmute(&uniforms as *const LinearUniforms) });
            encoder.set_buffer(0, Some(&digests_buffer), 0);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &tg_flags as *const u32 as *const core::ffi::c_void);

            let mut tg_mem = 0usize;
            if (tg_flags & TG_CONST_FLAG) != 0 {
                tg_mem += POSEIDON_CONST_SIZE;
            }
            if (tg_flags & TG_CHILD_CACHE_FLAG) != 0 {
                tg_mem += (num_threads as usize) * 8 * std::mem::size_of::<u64>();
            }
            encoder.set_threadgroup_memory_length(0, tg_mem as u64);

            let thread_group_size = MTLSize { width: num_threads, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
        }

        // Compute cap hashes
        {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_caps);
            encoder.set_buffer(0, Some(&caps_buffer), 0);
            encoder.set_buffer(1, Some(&digests_buffer), 0);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &tg_flags as *const u32 as *const core::ffi::c_void);

            // Use TrackedBuffer RAII for uniforms - auto-tracks on drop
            let uniforms_buf = self.alloc_with_data_tracked(&[uniforms]);
            encoder.set_buffer(2, Some(&*uniforms_buf), 0);

            let simd_width = pipeline_hash_caps.thread_execution_width() as usize;
            let max_threads = pipeline_hash_caps.max_total_threads_per_threadgroup() as usize;
            let num_threads = get_merkle_threadgroup_size(num_caps, simd_width, max_threads) as u64;
            let lg = (num_caps as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = MTLSize { width: lg.max(1), height: 1, depth: 1 };
            let thread_group_size = MTLSize { width: num_threads.min(num_caps as u64), height: 1, depth: 1 };

            let tg_mem = if (tg_flags & TG_CONST_FLAG) != 0 {
                POSEIDON_CONST_SIZE
            } else {
                0
            };
            encoder.set_threadgroup_memory_length(0, tg_mem as u64);

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            // uniforms_buf RAII: auto-tracks deallocation on drop
        }

        // Commit but DON'T wait - this is the key for async
        command_buffer.commit();

        // Return job handle with retained command buffer
        unsafe {
            MerkleGpuJob::new(
                command_buffer,
                digests_buffer,
                Some(caps_buffer),
                total_digests,
                num_caps,
                tree_height,
                cap_height,
                true, // is_linear_layout
            )
        }
    }
}
