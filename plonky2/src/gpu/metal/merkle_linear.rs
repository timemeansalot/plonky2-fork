//! Linear layout Merkle tree GPU implementation.
//!
//! Uses linear memory layout (zeknox pattern) for better memory coalescing on GPU.

#![cfg(any(feature = "metal-linear-merkle", feature = "metal-optimized"))]

use metal::{objc::rc::autoreleasepool, *};
use crate::{
    field::goldilocks_field::GoldilocksField,
    hash::{
        hash_types::HashOut,
        merkle_tree::{MerkleCap, MerkleTree},
        poseidon::PoseidonHash,
    },
};

use crate::gpu::metal::buffer_pool::get_buffer_pool_stats;
use crate::gpu::metal::runtime::MetalRuntime;
use crate::gpu::metal::threadgroup_config::get_merkle_threadgroup_size;
use crate::gpu::metal::tracking::track_deallocation;
use crate::gpu::metal::utils::{
    from_buf_raw, get_node_hash_index_in_digests, get_size_for_count, LinearUniforms,
};

impl MetalRuntime {
    /// Build a Merkle tree using linear memory layout (zeknox pattern)
    /// This provides better memory coalescing on GPU
    pub fn new_merkle_tree_linear(
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
                "[GPU Merkle Linear] Building tree: height={}, leaves={}, leaf_length={}, cap_height={}",
                tree_height, leaf_count, leaf_length, cap_height
            );
        }

        // Fall back to CPU for small trees or when cap_height equals tree_height
        if cap_height == tree_height || tree_height < 13 {
            if std::env::var("PLONKY2_METAL_LOG").is_ok() {
                println!("[GPU Merkle Linear] Using CPU fallback");
            }
            return MerkleTree::new_from_2d(leaves, cap_height);
        }

        // For very large trees (>=2^21), fall back to recursive GPU layout
        // The CPU convert overhead in linear layout dominates at this scale
        // Benchmarks show: recursive at 2^25 = 17.5s, linear at 2^25 = 17.9s (convert adds ~0.5-0.7s per tree)
        if tree_height >= 21 {
            if std::env::var("PLONKY2_METAL_LOG").is_ok() {
                println!(
                    "[GPU Merkle Linear] Tree height {} >= 21, using recursive layout (better for large trees)",
                    tree_height
                );
            }
            return Self::new_merkle_tree(leaves, cap_height);
        }

        if std::env::var("PLONKY2_METAL_LOG").is_ok() {
            println!("[GPU Merkle Linear] Using GPU with linear layout");
        }

        let leaf_u64s = leaves.concat();
        let (digests, caps) = autoreleasepool::<(Vec<HashOut<GoldilocksField>>, Vec<HashOut<GoldilocksField>>), _>(|| {
            // Use TrackedBuffer RAII for automatic tracking
            // Keep tracked until GPU work completes (hash function returns)
            let leaves_buffer = Self::get().alloc_with_data_tracked(&leaf_u64s);
            let result = Self::get().hash_merkle_tree_linear_buf_ho(
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

    /// Hash leaves only using the linear leaf kernel.
    /// This is a Poseidon kernel microbenchmark helper (no Merkle tree levels).
    pub fn poseidon_hash_leaves_linear_only(
        &self,
        leaves: &[GoldilocksField],
        leaf_size: usize,
    ) -> Vec<HashOut<GoldilocksField>> {
        let leaf_count = leaves.len() / leaf_size;
        assert!(leaf_count > 0, "leaf_count must be > 0");
        assert_eq!(leaves.len(), leaf_count * leaf_size, "leaves length must be multiple of leaf_size");

        let mut uniforms = LinearUniforms {
            level: 0,
            subtree_digests_len: leaf_count as u32,
            subtree_leaves_len: leaf_count as u32,
            leaf_size: leaf_size as u32,
            leaf_count: leaf_count as u32,
            subtree_count: 1,
            grid_width: 0,
        };

        let leaves_buffer = self.alloc_with_data_tracked(leaves);
        let buffer_size = leaf_count * 4 * std::mem::size_of::<u64>();
        let digests_buffer = self.alloc_aligned_tracked(buffer_size);

        let pipeline_hash_leaves = self.get_poseidon_hash_leaves_linear_pipeline_state();
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_pass_descriptor = ComputePassDescriptor::new();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

        encoder.set_compute_pipeline_state(pipeline_hash_leaves);
        encoder.set_buffer(0, Some(&*leaves_buffer), 0);
        encoder.set_buffer(1, Some(&*digests_buffer), 0);

        let simd_width = pipeline_hash_leaves.thread_execution_width() as usize;
        let max_threads = pipeline_hash_leaves.max_total_threads_per_threadgroup() as usize;
        let num_threads = get_merkle_threadgroup_size(leaf_count, simd_width, max_threads) as u64;

        let lg = (leaf_count as NSUInteger + num_threads - 1) / num_threads;
        let thread_group_count = get_size_for_count(lg as usize);
        let grid_width = (thread_group_count.width * num_threads) as u32;
        uniforms.grid_width = grid_width;

        encoder.set_bytes(2, std::mem::size_of::<LinearUniforms>() as u64, &uniforms as *const _ as *const core::ffi::c_void);

        let thread_group_size = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = digests_buffer.contents() as *mut HashOut<GoldilocksField>;
        let res = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(ptr, leaf_count) };

        res
    }

    /// Hash a single Merkle tree level (parent nodes) using the linear tree-level kernel.
    /// Inputs are raw child digests laid out as 8 * node_count u64s (two HashOuts per node).
    pub fn poseidon_hash_tree_level_linear_only(
        &self,
        child_digests: &[u64],
        node_count: usize,
    ) -> Vec<HashOut<GoldilocksField>> {
        assert_eq!(child_digests.len(), node_count * 8, "child_digests must be 8 * node_count");

        let subtree_leaves_len = 2 * node_count;
        let subtree_digests_len = 2 * subtree_leaves_len - 1;

        let mut uniforms = LinearUniforms {
            level: 1,
            subtree_digests_len: subtree_digests_len as u32,
            subtree_leaves_len: subtree_leaves_len as u32,
            leaf_size: 8,
            leaf_count: subtree_leaves_len as u32,
            subtree_count: 1,
            grid_width: 0,
        };

        let buffer_size = child_digests.len() * std::mem::size_of::<u64>();
        let output_size = subtree_digests_len * 4 * std::mem::size_of::<u64>();
        let output_buffer = self.alloc_aligned_tracked(output_size);

        let pipeline = self.get_poseidon_hash_tree_level_linear_pipeline_state();
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_pass_descriptor = ComputePassDescriptor::new();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&*output_buffer), 0);
        encoder.set_bytes(1, std::mem::size_of::<LinearUniforms>() as u64, &uniforms as *const _ as *const core::ffi::c_void);

        // Upload child digests into output buffer at the leaf region.
        // Linear layout expects leaves in the second half of subtree_digests_len.
        unsafe {
            let dst = output_buffer.contents() as *mut u8;
            let leaf_offset_bytes = ((subtree_digests_len - subtree_leaves_len) * 4 * std::mem::size_of::<u64>()) as isize;
            std::ptr::copy_nonoverlapping(
                child_digests.as_ptr() as *const u8,
                dst.offset(leaf_offset_bytes),
                buffer_size,
            );
        }

        let simd_width = pipeline.thread_execution_width() as usize;
        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
        let num_threads = get_merkle_threadgroup_size(node_count, simd_width, max_threads) as u64;

        let lg = (node_count as NSUInteger + num_threads - 1) / num_threads;
        let thread_group_count = get_size_for_count(lg as usize);
        let grid_width = (thread_group_count.width * num_threads) as u32;
        uniforms.grid_width = grid_width;

        encoder.set_bytes(1, std::mem::size_of::<LinearUniforms>() as u64, &uniforms as *const _ as *const core::ffi::c_void);

        let thread_group_size = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = output_buffer.contents() as *mut HashOut<GoldilocksField>;
        let res = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(ptr, node_count) };

        res
    }

    /// Hash merkle tree with linear layout, returning both digests and caps
    pub fn hash_merkle_tree_linear_buf_ho(
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
        let num_layers = tree_height - cap_height; // Number of levels in each subtree

        // Use TrackedBuffer RAII for digests - auto-tracks allocation/deallocation
        // Skip zeroing for pooled buffers - GPU writes all positions during leaf hashing
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

        // Get pipeline states
        let pipeline_hash_leaves = self.get_poseidon_hash_leaves_linear_pipeline_state();
        let pipeline_hash_tree_level = self.get_poseidon_hash_tree_level_linear_pipeline_state();
        let pipeline_hash_caps = self.get_poseidon_hash_caps_linear_pipeline_state();

        let command_buffer = self.command_queue.new_command_buffer();

        // Create uniforms (grid_width will be set per dispatch)
        let mut uniforms = LinearUniforms {
            level: 0,
            subtree_digests_len: subtree_digests_len as u32,
            subtree_leaves_len: subtree_leaves_len as u32,
            leaf_size: leaf_length as u32,
            leaf_count: leaf_count as u32,
            subtree_count: num_caps as u32,
            grid_width: 0, // Will be set before each dispatch
        };

        // Hash all leaves
        {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_leaves);
            encoder.set_buffer(0, Some(&leaves_buffer), 0);
            encoder.set_buffer(1, Some(&*digests_buffer), 0);

            // Use tuned threadgroup size, clamped to device limits
            let simd_width = pipeline_hash_leaves.thread_execution_width() as usize;
            let max_threads = pipeline_hash_leaves.max_total_threads_per_threadgroup() as usize;
            let num_threads = get_merkle_threadgroup_size(leaf_count, simd_width, max_threads) as u64;
            let lg = (leaf_count as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = get_size_for_count(lg as usize);

            // Calculate grid_width: total threads in X = thread_group_count.width * threads_per_group.width
            uniforms.grid_width = (thread_group_count.width * num_threads) as u32;

            // Use TrackedBuffer RAII for uniforms - auto-tracks on drop
            let uniforms_buffer = self.alloc_with_data_tracked(&[uniforms]);
            encoder.set_buffer(2, Some(&*uniforms_buffer), 0);

            let thread_group_size = MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            // uniforms_buffer RAII: auto-tracks deallocation on drop
        }

        // Hash internal tree levels
        // Process from level 1 (parents of leaves) up to num_layers - 1 (subtree root)
        for level in 1..num_layers {
            uniforms.level = level as u32;

            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_tree_level);
            encoder.set_buffer(0, Some(&*digests_buffer), 0);

            let nodes_at_this_level = subtree_leaves_len >> level;
            let total_nodes = nodes_at_this_level * num_caps;
            // Use tuned threadgroup size, clamped to device limits
            let simd_width = pipeline_hash_tree_level.thread_execution_width() as usize;
            let max_threads = pipeline_hash_tree_level.max_total_threads_per_threadgroup() as usize;
            let num_threads = get_merkle_threadgroup_size(leaf_count, simd_width, max_threads) as u64;
            let lg = (total_nodes as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = get_size_for_count(lg as usize);

            // Update grid_width for this level's dispatch configuration
            uniforms.grid_width = (thread_group_count.width * num_threads) as u32;

            // Pass uniforms directly as bytes
            encoder.set_bytes(
                1,
                std::mem::size_of::<LinearUniforms>() as u64,
                &uniforms as *const LinearUniforms as *const core::ffi::c_void,
            );

            let thread_group_size = MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
        }

        // Compute cap hashes
        // Cap hashes are computed from the top two nodes of each subtree
        {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_caps);
            encoder.set_buffer(0, Some(&*caps_buffer), 0);
            encoder.set_buffer(1, Some(&*digests_buffer), 0);

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

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            // uniforms_buffer RAII: auto-tracks deallocation on drop
        }

        let submit_start = std::time::Instant::now();
        command_buffer.commit();
        let commit_time = submit_start.elapsed();
        let gpu_start = std::time::Instant::now();
        command_buffer.wait_until_completed();
        let gpu_time = gpu_start.elapsed();

        // Convert linear layout digests to plonky2's expected format
        let convert_start = std::time::Instant::now();
        let digests = self.convert_linear_to_plonky2_digests(
            &*digests_buffer,
            tree_height,
            cap_height,
            subtree_digests_len,
            subtree_leaves_len,
        );
        let convert_time = convert_start.elapsed();

        // Read caps
        let caps_ptr = (*caps_buffer).contents() as *mut HashOut<GoldilocksField>;
        let caps = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(caps_ptr, num_caps) };

        // Debug timing for large trees
        if tree_height >= 20 {
            let (hits, misses, pooled) = get_buffer_pool_stats();
            eprintln!("[GPU Merkle Timing] height={}: commit={:.3}s, GPU wait={:.3}s, CPU convert={:.3}s, pool hits/misses={}/{}, pooled={:.1}MB",
                     tree_height, commit_time.as_secs_f64(), gpu_time.as_secs_f64(), convert_time.as_secs_f64(),
                     hits, misses, pooled as f64 / (1024.0 * 1024.0));
        }

        // Return TrackedBuffers to pool for reuse (tracks deallocation since buffer is no longer "in use")
        self.return_tracked_buffer(digests_buffer);
        self.return_tracked_buffer(caps_buffer);

        (digests, caps)
    }

    /// Convert digests from linear layout to plonky2's recursive layout format
    #[cfg(any(feature = "metal-linear-merkle", feature = "metal-optimized"))]
    pub(crate) fn convert_linear_to_plonky2_digests(
        &self,
        digests_buffer: &Buffer,
        tree_height: usize,
        cap_height: usize,
        subtree_digests_len: usize,
        subtree_leaves_len: usize,
    ) -> Vec<HashOut<GoldilocksField>> {
        let num_caps = 1usize << cap_height;
        let num_layers = tree_height - cap_height;
        let leaf_count = 1usize << tree_height;

        // Total digests in plonky2 format
        let total_tree_hashes = leaf_count * 2 - 1;
        let total_digests = total_tree_hashes - (num_caps * 2 - 1);
        let tree_length = total_digests >> cap_height;

        let mut result = vec![HashOut::default(); total_digests];

        // Read raw linear buffer
        let linear_ptr = digests_buffer.contents() as *const HashOut<GoldilocksField>;
        let linear_digests: &[HashOut<GoldilocksField>] = unsafe {
            std::slice::from_raw_parts(linear_ptr, subtree_digests_len * num_caps)
        };

        // Map from linear layout to recursive layout
        // For each subtree:
        for subtree_idx in 0..num_caps {
            let subtree_base = subtree_idx * subtree_digests_len;

            // Map leaves (level 0)
            for leaf_idx in 0..subtree_leaves_len {
                let linear_idx = subtree_base + (subtree_digests_len - subtree_leaves_len) + leaf_idx;
                let global_leaf_idx = subtree_idx * subtree_leaves_len + leaf_idx;
                let recursive_idx = get_node_hash_index_in_digests(num_layers, tree_length, 0, global_leaf_idx);
                result[recursive_idx] = linear_digests[linear_idx];
            }

            // Map internal nodes (level 1 to num_layers-1)
            for level in 1..num_layers {
                let nodes_at_level = subtree_leaves_len >> level;
                for node_idx in 0..nodes_at_level {
                    // Linear index: level_start + node_idx within subtree
                    // Simplified formula to avoid unsigned underflow:
                    // level_start = (subtree_leaves_len >> level) - 1
                    // This is equivalent to: subtree_digests_len - 2*subtree_leaves_len + (subtree_leaves_len >> level)
                    let level_start = (subtree_leaves_len >> level) - 1;
                    let linear_idx = subtree_base + level_start + node_idx;

                    // Recursive index
                    let global_node_idx = subtree_idx * nodes_at_level + node_idx;
                    let recursive_idx = get_node_hash_index_in_digests(num_layers, tree_length, level, global_node_idx);
                    result[recursive_idx] = linear_digests[linear_idx];
                }
            }
        }

        result
    }
}
