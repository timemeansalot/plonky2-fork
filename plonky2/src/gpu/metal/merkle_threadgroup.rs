//! Threadgroup-optimized Merkle tree GPU implementation.
//!
//! Uses threadgroup memory for caching Poseidon round constants and MDS matrix.

#![cfg(feature = "metal-threadgroup")]

use metal::{objc::rc::autoreleasepool, *};
use crate::{
    field::goldilocks_field::GoldilocksField,
    hash::{
        hash_types::HashOut,
        merkle_tree::{MerkleCap, MerkleTree},
        poseidon::PoseidonHash,
    },
    plonk::config::Hasher,
};

use crate::gpu::metal::runtime::MetalRuntime;
use crate::gpu::metal::threadgroup_config::get_merkle_threadgroup_size;
use crate::gpu::metal::tracking::track_deallocation;
use crate::gpu::metal::utils::{from_buf_raw, get_node_hash_index_in_digests, get_size_for_count, POSEIDON_CONST_SIZE};

impl MetalRuntime {
    /// Build a Merkle tree using threadgroup-optimized kernels
    pub fn new_merkle_tree_threadgroup(
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
                "[GPU Merkle Threadgroup] Building tree: height={}, leaves={}, leaf_length={}, cap_height={}",
                tree_height, leaf_count, leaf_length, cap_height
            );
        }

        if cap_height == tree_height || tree_height < 13 {
            if std::env::var("PLONKY2_METAL_LOG").is_ok() {
                println!("[GPU Merkle Threadgroup] Using CPU fallback");
            }
            return MerkleTree::new_from_2d(leaves, cap_height);
        }
        if std::env::var("PLONKY2_METAL_LOG").is_ok() {
            println!("[GPU Merkle Threadgroup] Using GPU with threadgroup optimization");
        }

        let num_caps = 1usize << cap_height;
        let total_tree_hashes = leaf_count * 2 - 1;

        let total_digests = total_tree_hashes - (num_caps * 2 - 1);
        let num_layers = tree_height - cap_height;
        let tree_length = total_digests >> cap_height;

        let leaf_u64s = leaves.concat();
        let digests = autoreleasepool::<Vec<HashOut<GoldilocksField>>,_>(|| {
            // Use TrackedBuffer RAII for automatic tracking
            // Keep tracked until GPU work completes (hash function returns)
            let leaves_buffer = Self::get().alloc_with_data_tracked(&leaf_u64s);
            let result = Self::get().hash_merkle_tree_threadgroup_buf_ho(
                leaves_buffer.into_inner_untracked(),
                tree_height,
                leaf_length,
                cap_height,
            );
            // Now GPU work is complete, track deallocation
            track_deallocation(leaf_u64s.len() * std::mem::size_of::<GoldilocksField>());
            result
        });

        let mut caps: Vec<HashOut<GoldilocksField>> = Vec::with_capacity(num_caps);
        let cap_child_level = tree_height - cap_height - 1;
        for i in 0..num_caps {
            let left_index =
                get_node_hash_index_in_digests(num_layers, tree_length, cap_child_level, i * 2);
            caps.push(<PoseidonHash as Hasher<GoldilocksField>>::two_to_one(
                digests[left_index],
                digests[left_index + 1],
            ));
        }

        MerkleTree::new_from_fields(leaf_u64s, leaf_length, digests, MerkleCap(caps))
    }

    /// Hash merkle tree with threadgroup optimization
    pub fn hash_merkle_tree_threadgroup_buf_ho(
        &self,
        leaves_buffer: Buffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
    ) -> Vec<HashOut<GoldilocksField>> {
        let leaf_count = 1usize << tree_height;

        assert!(
            cap_height < (tree_height as usize),
            "cap height must be less than tree height"
        );
        let num_caps = 1usize << cap_height;
        let total_tree_hashes = leaf_count * 2 - 1;

        let total_digests = total_tree_hashes - (num_caps * 2 - 1);
        let num_layers = tree_height - cap_height;
        let tree_length = total_digests >> cap_height;

        let mut uniforms: [u32; 6] = [0, tree_length as u32, num_layers as u32, leaf_length as u32, leaf_count as u32, 0];

        // Use TrackedBuffer RAII for digests - auto-tracks allocation/deallocation
        let buffer_size = (total_digests * 4) * std::mem::size_of::<u64>();
        let digests_buffer = self.alloc_aligned_tracked(buffer_size);

        unsafe {
            let ptr = digests_buffer.contents() as *mut u8;
            std::ptr::write_bytes(ptr, 0, Self::align_to_256(buffer_size));
        }

        let pipeline_hash_leaves = self.get_poseidon_hash_leaves_threadgroup_pipeline_state();
        let pipeline_hash_tree_level = self.get_poseidon_hash_tree_level_threadgroup_pipeline_state();

        let command_buffer = self.command_queue.new_command_buffer();

        // Hash leaves with threadgroup-cached round constants and MDS constants
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

            let grid_width = (thread_group_count.width * num_threads) as u32;
            uniforms[5] = grid_width;

            // Use TrackedBuffer RAII for uniforms - auto-tracks on drop
            let uniforms_buffer = self.alloc_with_data_tracked(&uniforms);
            encoder.set_buffer(2, Some(&*uniforms_buffer), 0);

            // Set threadgroup memory for round constants + MDS constants (leaves don't need child cache)
            encoder.set_threadgroup_memory_length(0, POSEIDON_CONST_SIZE as u64);

            let thread_group_size = MTLSize {
                width: num_threads,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            // uniforms_buffer RAII: auto-tracks deallocation on drop
        }

        // Hash tree levels with threadgroup memory for both constants and child cache
        for i in 1..num_layers {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder = command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(&pipeline_hash_tree_level);

            uniforms[0] = i as u32;

            // Use tuned threadgroup size, clamped to device limits
            let simd_width = pipeline_hash_tree_level.thread_execution_width() as usize;
            let max_threads = pipeline_hash_tree_level.max_total_threads_per_threadgroup() as usize;
            let num_threads = get_merkle_threadgroup_size(leaf_count, simd_width, max_threads) as u64;
            let lg = ((leaf_count >> i) as NSUInteger + num_threads - 1) / num_threads;
            let thread_group_count = get_size_for_count(lg as usize);

            let grid_width = (thread_group_count.width * num_threads) as u32;
            uniforms[5] = grid_width;

            encoder.set_bytes(1, 24, uniforms.as_ptr() as *mut core::ffi::c_void);
            encoder.set_buffer(0, Some(&*digests_buffer), 0);

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

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = digests_buffer.contents() as *mut HashOut<GoldilocksField>;
        let res = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(ptr, total_digests) };

        // TrackedBuffer RAII: digests_buffer auto-tracks deallocation on drop
        res
    }
}
