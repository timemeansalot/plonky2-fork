//! GPU Merkle tree construction using linear+threadgroup Metal shader.

use metal::*;
use plonky2_field::goldilocks_field::GoldilocksField;

use crate::hash::hash_types::HashOut;
use crate::hash::metal::runtime::MetalRuntime;
use crate::hash::metal::threadgroup_config::get_merkle_threadgroup_size;
use crate::hash::metal::utils::{
    from_buf_raw, get_node_hash_index_in_digests, get_size_for_count, LinearUniforms,
    POSEIDON_CONST_SIZE,
};

impl MetalRuntime {
    /// Hash merkle tree with linear layout + threadgroup optimization, returning digests and caps.
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
        let (digests_buffer, digests_from_pool) =
            self.alloc_aligned_tracked_with_pool_hint(buffer_size);

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
        let pipeline_hash_leaves =
            self.get_poseidon_hash_leaves_linear_threadgroup_pipeline_state();
        let pipeline_hash_tree_level =
            self.get_poseidon_hash_tree_level_linear_threadgroup_pipeline_state();
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

        // Hash all leaves
        {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_leaves);
            encoder.set_buffer(0, Some(&leaves_buffer), 0);
            encoder.set_buffer(1, Some(&*digests_buffer), 0);

            // Use tuned threadgroup size based on tree size, clamped to device limits
            let simd_width = pipeline_hash_leaves.thread_execution_width() as usize;
            let max_threads =
                pipeline_hash_leaves.max_total_threads_per_threadgroup() as usize;
            let num_threads =
                get_merkle_threadgroup_size(leaf_count, simd_width, max_threads) as u64;
            let lg = (leaf_count as NSUInteger + num_threads - 1) / num_threads;
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

            // Set threadgroup memory for round constants + MDS constants
            encoder.set_threadgroup_memory_length(0, POSEIDON_CONST_SIZE as u64);

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            // uniforms_buffer RAII: auto-tracks deallocation on drop
        }

        // Hash internal tree levels with threadgroup memory
        for level in 1..num_layers {
            uniforms.level = level as u32;

            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_tree_level);
            encoder.set_buffer(0, Some(&*digests_buffer), 0);

            let nodes_at_this_level = subtree_leaves_len >> level;
            let total_nodes = nodes_at_this_level * num_caps;
            // Use tuned threadgroup size, clamped to device limits
            let simd_width = pipeline_hash_tree_level.thread_execution_width() as usize;
            let max_threads =
                pipeline_hash_tree_level.max_total_threads_per_threadgroup() as usize;
            let num_threads =
                get_merkle_threadgroup_size(leaf_count, simd_width, max_threads) as u64;
            let lg = (total_nodes as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = get_size_for_count(lg as usize);

            uniforms.grid_width = (thread_group_count.width * num_threads) as u32;

            // Pass uniforms as bytes
            encoder.set_bytes(
                1,
                std::mem::size_of::<LinearUniforms>() as u64,
                &uniforms as *const LinearUniforms as *const core::ffi::c_void,
            );

            // Set threadgroup memory: round constants + MDS constants + child cache (8 ulongs per thread)
            let child_cache_size = (num_threads as usize) * 8 * std::mem::size_of::<u64>();
            let threadgroup_mem_size = POSEIDON_CONST_SIZE + child_cache_size;
            encoder.set_threadgroup_memory_length(0, threadgroup_mem_size as u64);

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
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_caps);
            encoder.set_buffer(0, Some(&*caps_buffer), 0);
            encoder.set_buffer(1, Some(&*digests_buffer), 0);

            // Use TrackedBuffer RAII for uniforms - auto-tracks on drop
            let uniforms_buffer = self.alloc_with_data_tracked(&[uniforms]);
            encoder.set_buffer(2, Some(&*uniforms_buffer), 0);

            // Use tuned threadgroup size, clamped to device limits
            let simd_width = pipeline_hash_caps.thread_execution_width() as usize;
            let max_threads =
                pipeline_hash_caps.max_total_threads_per_threadgroup() as usize;
            let num_threads =
                get_merkle_threadgroup_size(num_caps, simd_width, max_threads) as u64;
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

            // Set threadgroup memory for round constants + MDS constants
            encoder.set_threadgroup_memory_length(0, POSEIDON_CONST_SIZE as u64);

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

        // Return TrackedBuffers to pool for reuse
        self.return_tracked_buffer(digests_buffer);
        self.return_tracked_buffer(caps_buffer);

        (digests, caps)
    }

    /// Convert digests from linear layout to plonky2's recursive layout format.
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

        // Map from linear layout to recursive layout; for each subtree:
        for subtree_idx in 0..num_caps {
            let subtree_base = subtree_idx * subtree_digests_len;

            // Map leaves (level 0)
            for leaf_idx in 0..subtree_leaves_len {
                let linear_idx =
                    subtree_base + (subtree_digests_len - subtree_leaves_len) + leaf_idx;
                let global_leaf_idx = subtree_idx * subtree_leaves_len + leaf_idx;
                let recursive_idx =
                    get_node_hash_index_in_digests(num_layers, tree_length, 0, global_leaf_idx);
                result[recursive_idx] = linear_digests[linear_idx];
            }

            // Map internal nodes (level 1 to num_layers-1)
            for level in 1..num_layers {
                let nodes_at_level = subtree_leaves_len >> level;
                for node_idx in 0..nodes_at_level {
                    // level_start avoids unsigned underflow by using simplified form
                    let level_start = (subtree_leaves_len >> level) - 1;
                    let linear_idx = subtree_base + level_start + node_idx;

                    let global_node_idx = subtree_idx * nodes_at_level + node_idx;
                    let recursive_idx = get_node_hash_index_in_digests(
                        num_layers,
                        tree_length,
                        level,
                        global_node_idx,
                    );
                    result[recursive_idx] = linear_digests[linear_idx];
                }
            }
        }

        result
    }
}
