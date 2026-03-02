//! GPU Merkle tree construction using Metal shaders (linear+threadgroup and coalesced).

use metal::*;
use plonky2_field::goldilocks_field::GoldilocksField;

use crate::hash::hash_types::HashOut;
use crate::hash::metal::runtime::MetalRuntime;
use crate::hash::metal::threadgroup_config::get_merkle_threadgroup_size;
use crate::hash::metal::utils::{
    from_buf_raw, get_size_for_count, CoalescedUniforms, LinearUniforms, POSEIDON_CONST_SIZE,
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

        #[cfg(feature = "timing")]
        let gt0 = std::time::Instant::now();

        let num_caps = 1usize << cap_height;
        let subtree_leaves_len = leaf_count >> cap_height;
        // Use plonky2-fork's BFS layout: 2*(n-1) digests per subtree (no root slot).
        // The shader index formulas produce BFS positions when given this value:
        //   leaf:     (subtree_digests_len - n) + leaf_idx = (n - 2) + leaf_idx
        //   internal: subtree_digests_len - 2*n + (n >> level) = (n >> level) - 2
        // This matches plonky2-fork's layout, eliminating the need for post-hoc conversion.
        let subtree_digests_len = 2 * (subtree_leaves_len - 1);
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

        #[cfg(feature = "timing")]
        let gt1 = std::time::Instant::now();

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

        #[cfg(feature = "timing")]
        let gt2 = std::time::Instant::now();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        #[cfg(feature = "timing")]
        let gt3 = std::time::Instant::now();

        // GPU wrote directly in plonky2-fork BFS layout — read directly, no conversion needed.
        let digests_ptr = (*digests_buffer).contents() as *const HashOut<GoldilocksField>;
        let digests = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(digests_ptr, total_digests) };

        // Read caps
        let caps_ptr = (*caps_buffer).contents() as *mut HashOut<GoldilocksField>;
        let caps = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(caps_ptr, num_caps) };

        // Return TrackedBuffers to pool for reuse
        self.return_tracked_buffer(digests_buffer);
        self.return_tracked_buffer(caps_buffer);

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

        (digests, caps)
    }

    /// Hash merkle tree with coalesced memory access pattern, returning digests and caps.
    ///
    /// Uses 2D dispatch (X = node within subtree, Y = subtree index) for guaranteed
    /// memory coalescing. All three phases (leaves, internal levels, caps) are encoded
    /// into a single command buffer to minimize overhead.
    ///
    /// Best for large trees (>= 2^21 leaves) where memory bandwidth is the bottleneck.
    pub fn hash_merkle_tree_coalesced_buf_ho(
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

        #[cfg(feature = "timing")]
        let gt0 = std::time::Instant::now();

        let num_caps = 1usize << cap_height;
        let subtree_leaves_len = leaf_count >> cap_height;
        // Use plonky2-fork's BFS layout directly (same as linear_threadgroup path).
        let subtree_digests_len = 2 * (subtree_leaves_len - 1);
        let total_digests = subtree_digests_len * num_caps;
        let num_layers = tree_height - cap_height;

        // Allocate digests buffer with pool hint
        let buffer_size = (total_digests * 4) * std::mem::size_of::<u64>();
        let (digests_buffer, digests_from_pool) =
            self.alloc_aligned_tracked_with_pool_hint(buffer_size);

        if !digests_from_pool {
            unsafe {
                let ptr = digests_buffer.contents() as *mut u8;
                std::ptr::write_bytes(ptr, 0, Self::align_to_256(buffer_size));
            }
        }

        // Allocate caps buffer with pool hint
        let caps_size = (num_caps * 4) * std::mem::size_of::<u64>();
        let (caps_buffer, caps_from_pool) = self.alloc_aligned_tracked_with_pool_hint(caps_size);
        if !caps_from_pool {
            unsafe {
                let ptr = caps_buffer.contents() as *mut u8;
                std::ptr::write_bytes(ptr, 0, Self::align_to_256(caps_size));
            }
        }

        #[cfg(feature = "timing")]
        let gt1 = std::time::Instant::now();

        // Get coalesced pipeline states
        let pipeline_hash_leaves = self.get_poseidon_hash_leaves_coalesced_pipeline_state();
        let pipeline_hash_tree_level =
            self.get_poseidon_hash_tree_level_coalesced_pipeline_state();
        let pipeline_hash_caps = self.get_poseidon_hash_caps_coalesced_pipeline_state();

        // Single command buffer for all 3 phases
        let command_buffer = self.command_queue.new_command_buffer();

        // Phase 1: Hash all leaves with 2D dispatch
        {
            let simd_width = pipeline_hash_leaves.thread_execution_width() as usize;
            let max_threads =
                pipeline_hash_leaves.max_total_threads_per_threadgroup() as usize;

            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(pipeline_hash_leaves);
            encoder.set_buffer(0, Some(&leaves_buffer), 0);
            encoder.set_buffer(1, Some(&*digests_buffer), 0);

            // 2D grid: X = leaf within subtree, Y = subtree index
            let grid_size = MTLSize {
                width: subtree_leaves_len as u64,
                height: num_caps as u64,
                depth: 1,
            };

            let uniforms = CoalescedUniforms {
                level: 0,
                subtree_digests_len: subtree_digests_len as u32,
                subtree_leaves_len: subtree_leaves_len as u32,
                leaf_size: leaf_length as u32,
                leaf_count: leaf_count as u32,
                subtree_count: num_caps as u32,
                nodes_per_subtree: subtree_leaves_len as u32,
                nodes_per_dispatch: leaf_count as u32,
                dispatch_offset: 0,
                enable_counters: 0,
            };
            encoder.set_bytes(
                2,
                std::mem::size_of::<CoalescedUniforms>() as u64,
                &uniforms as *const CoalescedUniforms as *const core::ffi::c_void,
            );

            let threadgroup_size =
                get_merkle_threadgroup_size(subtree_leaves_len, simd_width, max_threads);
            encoder.set_threadgroup_memory_length(0, POSEIDON_CONST_SIZE as u64);

            let tg_size = MTLSize {
                width: threadgroup_size as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_threads(grid_size, tg_size);
            encoder.end_encoding();
        }

        // Phase 2: Hash internal levels with 2D dispatch
        {
            let simd_width = pipeline_hash_tree_level.thread_execution_width() as usize;
            let max_threads =
                pipeline_hash_tree_level.max_total_threads_per_threadgroup() as usize;

            for level in 1..num_layers {
                let nodes_per_subtree = subtree_leaves_len >> level;

                let compute_pass_descriptor = ComputePassDescriptor::new();
                let encoder = command_buffer
                    .compute_command_encoder_with_descriptor(compute_pass_descriptor);

                encoder.set_compute_pipeline_state(pipeline_hash_tree_level);
                encoder.set_buffer(0, Some(&*digests_buffer), 0);

                // 2D grid: X = node within subtree at this level, Y = subtree index
                let grid_size = MTLSize {
                    width: nodes_per_subtree as u64,
                    height: num_caps as u64,
                    depth: 1,
                };

                let uniforms = CoalescedUniforms {
                    level: level as u32,
                    subtree_digests_len: subtree_digests_len as u32,
                    subtree_leaves_len: subtree_leaves_len as u32,
                    leaf_size: 0,
                    leaf_count: 0,
                    subtree_count: num_caps as u32,
                    nodes_per_subtree: nodes_per_subtree as u32,
                    nodes_per_dispatch: (nodes_per_subtree * num_caps) as u32,
                    dispatch_offset: 0,
                    enable_counters: 0,
                };
                encoder.set_bytes(
                    1,
                    std::mem::size_of::<CoalescedUniforms>() as u64,
                    &uniforms as *const CoalescedUniforms as *const core::ffi::c_void,
                );

                let threadgroup_size =
                    get_merkle_threadgroup_size(nodes_per_subtree, simd_width, max_threads);
                let child_cache_size =
                    (threadgroup_size as usize) * 8 * std::mem::size_of::<u64>();
                let threadgroup_memory_size = POSEIDON_CONST_SIZE + child_cache_size;
                encoder.set_threadgroup_memory_length(0, threadgroup_memory_size as u64);

                let tg_size = MTLSize {
                    width: threadgroup_size as u64,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatch_threads(grid_size, tg_size);
                encoder.end_encoding();
            }
        }

        // Phase 3: Compute cap hashes
        {
            let simd_width = pipeline_hash_caps.thread_execution_width() as usize;
            let max_threads =
                pipeline_hash_caps.max_total_threads_per_threadgroup() as usize;

            let compute_pass_descriptor = ComputePassDescriptor::new();
            let encoder =
                command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

            encoder.set_compute_pipeline_state(pipeline_hash_caps);
            encoder.set_buffer(0, Some(&*caps_buffer), 0);
            encoder.set_buffer(1, Some(&*digests_buffer), 0);

            let threadgroup_size =
                get_merkle_threadgroup_size(num_caps, simd_width, max_threads);

            let uniforms = CoalescedUniforms {
                level: 0,
                subtree_digests_len: subtree_digests_len as u32,
                subtree_leaves_len: subtree_leaves_len as u32,
                leaf_size: leaf_length as u32,
                leaf_count: leaf_count as u32,
                subtree_count: num_caps as u32,
                nodes_per_subtree: 1,
                nodes_per_dispatch: num_caps as u32,
                dispatch_offset: 0,
                enable_counters: 0,
            };
            encoder.set_bytes(
                2,
                std::mem::size_of::<CoalescedUniforms>() as u64,
                &uniforms as *const CoalescedUniforms as *const core::ffi::c_void,
            );

            encoder.set_threadgroup_memory_length(0, POSEIDON_CONST_SIZE as u64);

            let grid_size = MTLSize {
                width: num_caps as u64,
                height: 1,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: threadgroup_size as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_threads(grid_size, tg_size);
            encoder.end_encoding();
        }

        #[cfg(feature = "timing")]
        let gt2 = std::time::Instant::now();

        // Single commit and wait for all work
        command_buffer.commit();
        command_buffer.wait_until_completed();

        #[cfg(feature = "timing")]
        let gt3 = std::time::Instant::now();

        // GPU wrote directly in plonky2-fork BFS layout — read directly, no conversion needed.
        let digests_ptr = (*digests_buffer).contents() as *const HashOut<GoldilocksField>;
        let digests = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(digests_ptr, total_digests) };

        // Read caps
        let caps_ptr = (*caps_buffer).contents() as *mut HashOut<GoldilocksField>;
        let caps = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(caps_ptr, num_caps) };

        // Return TrackedBuffers to pool for reuse
        self.return_tracked_buffer(digests_buffer);
        self.return_tracked_buffer(caps_buffer);

        #[cfg(feature = "timing")]
        {
            let gt4 = std::time::Instant::now();
            eprintln!(
                "[merkle-gpu-inner h={} coalesced] alloc={:.1}ms encode={:.1}ms execute={:.1}ms readback={:.1}ms total={:.1}ms",
                tree_height,
                gt1.duration_since(gt0).as_secs_f64() * 1000.0,
                gt2.duration_since(gt1).as_secs_f64() * 1000.0,
                gt3.duration_since(gt2).as_secs_f64() * 1000.0,
                gt4.duration_since(gt3).as_secs_f64() * 1000.0,
                gt4.duration_since(gt0).as_secs_f64() * 1000.0,
            );
        }

        (digests, caps)
    }

}
