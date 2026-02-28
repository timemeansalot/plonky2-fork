//! Utility types and functions for Metal GPU Merkle tree implementation.

use metal::MTLSize;
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

/// Uniforms for coalesced bandwidth-optimized kernels.
/// Must match `CoalescedUniforms` in `poseidon_merkle_hasher_coalesced.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CoalescedUniforms {
    pub level: u32,
    pub subtree_digests_len: u32,
    pub subtree_leaves_len: u32,
    pub leaf_size: u32,
    pub leaf_count: u32,
    pub subtree_count: u32,
    pub nodes_per_subtree: u32,
    pub nodes_per_dispatch: u32,
    pub dispatch_offset: u32,
    pub enable_counters: u32,
}

/// Poseidon threadgroup memory constants (must match shader).
pub const POSEIDON_RC_SIZE: usize = 360 * std::mem::size_of::<u64>(); // 2880 bytes
pub const POSEIDON_MDS_SIZE: usize = 12 * std::mem::size_of::<i64>(); // 96 bytes
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
        MTLSize {
            width: count as u64,
            height: 1,
            depth: 1,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coalesced_uniforms_size() {
        assert_eq!(std::mem::size_of::<CoalescedUniforms>(), 40);
    }

    #[test]
    fn test_linear_uniforms_size() {
        assert_eq!(std::mem::size_of::<LinearUniforms>(), 28);
    }
}
