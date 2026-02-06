//! Utility functions for Merkle tree GPU operations.
//!
//! Includes layout conversion, size calculations, and memory helpers.

use metal::MTLSize;
use crate::{field::goldilocks_field::GoldilocksField, hash::hash_types::HashOut};
use std::ptr;

/// Linear layout uniforms struct - must match Metal shader
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LinearUniforms {
    pub level: u32,               // Current tree level being processed
    pub subtree_digests_len: u32, // Number of digests per subtree
    pub subtree_leaves_len: u32,  // Number of leaves per subtree
    pub leaf_size: u32,           // Elements per leaf
    pub leaf_count: u32,          // Total number of leaves
    pub subtree_count: u32,       // Number of subtrees
    pub grid_width: u32,          // Width of the dispatch grid (threads in X dimension)
}

/// Coalesced layout uniforms struct - extends LinearUniforms with 2D dispatch support
/// Must match CoalescedUniforms in poseidon_merkle_hasher_coalesced.metal
/// Uses 2D dispatch: X = node within subtree, Y = subtree index
/// This guarantees all threads in a threadgroup are in the same subtree
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CoalescedUniforms {
    pub level: u32,               // Current tree level being processed
    pub subtree_digests_len: u32, // Number of digests per subtree
    pub subtree_leaves_len: u32,  // Number of leaves per subtree
    pub leaf_size: u32,           // Elements per leaf
    pub leaf_count: u32,          // Total number of leaves
    pub subtree_count: u32,       // Number of subtrees
    pub nodes_per_subtree: u32,   // Nodes per subtree at this level (for 2D dispatch)
    pub nodes_per_dispatch: u32,  // Number of nodes to process in this dispatch
    pub dispatch_offset: u32,     // Starting node offset for this dispatch
    pub enable_counters: u32,     // Non-zero to enable sequential access counters
}

/// Sequential access counters for instrumentation
/// Must match AccessCounters in poseidon_merkle_hasher_coalesced.metal
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AccessCounters {
    pub total_accesses: u32,      // Total number of memory accesses
    pub sequential_accesses: u32, // Number of sequential (coalesced) accesses
}

/// Recursive layout indexing (original implementation)
pub fn get_node_hash_index_in_digests(
    num_layers: usize,
    tree_length: usize,
    level: usize,
    index: usize,
) -> usize {
    let leaf_index = index << level;
    let tree_index = index >> (num_layers - level);
    let pair_index = ((leaf_index) & ((1 << num_layers) - 1)) >> level;
    let parity = pair_index & 1;
    let pair_index = pair_index >> 1;

    let siblings_index = (pair_index << (level + 1)) + (1 << level) - 1;
    let d_index = 2 * siblings_index + parity;
    d_index + (tree_index * tree_length)
}

/// Convert linear layout digests to plonky2's recursive layout format
/// This is a standalone version for use by MerkleGpuJob::finish()
pub fn convert_linear_to_recursive(
    linear_digests: &[HashOut<GoldilocksField>],
    num_layers: usize,
    tree_length: usize,
    num_caps: usize,
) -> Vec<HashOut<GoldilocksField>> {
    let subtree_leaves_len = 1usize << (num_layers);
    let subtree_digests_len = 2 * subtree_leaves_len - 1;
    let total_digests = tree_length * num_caps;

    let mut result = vec![HashOut::default(); total_digests];

    // Map from linear layout to recursive layout
    for subtree_idx in 0..num_caps {
        let subtree_base = subtree_idx * subtree_digests_len;

        // Map leaves (level 0)
        for leaf_idx in 0..subtree_leaves_len {
            let linear_idx = subtree_base + (subtree_digests_len - subtree_leaves_len) + leaf_idx;
            let global_leaf_idx = subtree_idx * subtree_leaves_len + leaf_idx;
            let recursive_idx = get_node_hash_index_in_digests(num_layers, tree_length, 0, global_leaf_idx);
            if linear_idx < linear_digests.len() && recursive_idx < result.len() {
                result[recursive_idx] = linear_digests[linear_idx];
            }
        }

        // Map internal nodes (level 1 to num_layers-1)
        for level in 1..num_layers {
            let nodes_at_level = subtree_leaves_len >> level;
            for node_idx in 0..nodes_at_level {
                let level_start = (subtree_leaves_len >> level) - 1;
                let linear_idx = subtree_base + level_start + node_idx;
                let global_node_idx = subtree_idx * nodes_at_level + node_idx;
                let recursive_idx = get_node_hash_index_in_digests(num_layers, tree_length, level, global_node_idx);
                if linear_idx < linear_digests.len() && recursive_idx < result.len() {
                    result[recursive_idx] = linear_digests[linear_idx];
                }
            }
        }
    }

    result
}

/// Copy raw data from a pointer to a Vec
pub unsafe fn from_buf_raw<T>(ptr: *const T, elts: usize) -> Vec<T> {
    let mut dst = Vec::with_capacity(elts);

    // SAFETY: Our precondition ensures the source is aligned and valid,
    // and `Vec::with_capacity` ensures that we have usable space to write them.
    ptr::copy(ptr, dst.as_mut_ptr(), elts);

    // SAFETY: We created it with this much capacity earlier,
    // and the previous `copy` has initialized these elements.
    dst.set_len(elts);
    dst
}

/// Get MTLSize for a given thread count, using 2D dispatch for large counts
pub fn get_size_for_count(count: usize) -> MTLSize {
    let max_dim = 32768usize;
    if count <= max_dim {
        MTLSize {
            width: count as u64,
            height: 1,
            depth: 1,
        }
    } else {
        // Round up to avoid losing threads due to integer division
        MTLSize {
            width: max_dim as u64,
            height: ((count + max_dim - 1) / max_dim) as u64,
            depth: 1,
        }
    }
}

/// Poseidon threadgroup memory constants
pub const POSEIDON_RC_SIZE: usize = 360 * std::mem::size_of::<u64>(); // 2880 bytes
pub const POSEIDON_MDS_SIZE: usize = 12 * std::mem::size_of::<i64>(); // 96 bytes
pub const POSEIDON_CONST_SIZE: usize = POSEIDON_RC_SIZE + POSEIDON_MDS_SIZE; // 2976 bytes
