// Linear indexing Merkle tree implementation
// Based on zeknox CUDA implementation pattern for better memory coalescing
//
// Memory Layout (for tree with 4 leaves):
// Linear Array:
// Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]
// Value:  H0123 H01   H23   L0    L1    L2    L3
//         (root)(lvl1)(lvl1)(leaf)(leaf)(leaf)(leaf)
//
// For subtree at index subtree_idx with subtree_digests_len total digests:
// - Leaves are stored in the second half: offset = subtree_digests_len - subtree_leaves_len
// - Internal nodes are stored in the first half, level by level from root to leaves-1

#include <metal_stdlib>
#include "goldilocks.metal"
#include "poseidon_goldilocks.metal"

using namespace metal;
using namespace GoldilocksField;

// Uniforms for linear layout
struct LinearUniforms {
    uint level;               // Current tree level being processed (0 = leaf level)
    uint subtree_digests_len; // Number of digests per subtree (2 * leaves - 1)
    uint subtree_leaves_len;  // Number of leaves per subtree
    uint leaf_size;           // Elements per leaf
    uint leaf_count;          // Total number of leaves
    uint subtree_count;       // Number of subtrees (2^cap_height)
    uint grid_width;          // Width of the dispatch grid (threads in X dimension)
};

// Compute digest index for a leaf in linear layout
// Leaves are stored in the second half of each subtree's digest array
inline uint compute_linear_leaf_index(
    uint subtree_digests_len,
    uint subtree_leaves_len,
    uint subtree_idx,
    uint in_subtree_idx
) {
    // Leaves start at offset (subtree_digests_len - subtree_leaves_len) within each subtree
    return subtree_idx * subtree_digests_len
         + (subtree_digests_len - subtree_leaves_len)
         + in_subtree_idx;
}

// Compute digest index for an internal node in linear layout
// Internal nodes are stored by level, from last level (above leaves) toward root
inline uint compute_linear_internal_index(
    uint subtree_digests_len,
    uint subtree_leaves_len,
    uint subtree_idx,
    uint level,          // Level counting from leaves: level 1 = parents of leaves
    uint index_in_level  // Index within this level
) {
    // For level i (counting from leaves), there are (subtree_leaves_len >> i) nodes
    // Level 1 nodes start at offset: subtree_digests_len - subtree_leaves_len - (subtree_leaves_len >> 1)
    // Level 2 nodes start at: above - (subtree_leaves_len >> 2)
    // etc.

    // Calculate offset from the start of subtree
    // Leaves are at: subtree_digests_len - subtree_leaves_len
    // Level 1 is at: subtree_digests_len - subtree_leaves_len - (subtree_leaves_len >> 1)
    // Level k is at: subtree_digests_len - subtree_leaves_len - sum(i=1 to k of (subtree_leaves_len >> i))
    //             = subtree_digests_len - subtree_leaves_len - (subtree_leaves_len - (subtree_leaves_len >> k))
    //             = subtree_digests_len - 2*subtree_leaves_len + (subtree_leaves_len >> level)

    uint level_start = subtree_digests_len - 2 * subtree_leaves_len + (subtree_leaves_len >> level);
    return subtree_idx * subtree_digests_len + level_start + index_in_level;
}

// Kernel to hash leaves with linear layout
kernel void poseidon_hash_leaves_linear(
    constant Fp * leaf_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
    constant LinearUniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];

    // Bounds check: don't process threads beyond the actual leaf count
    if (thread_id >= uniforms.leaf_count) {
        return;
    }

    // Calculate which subtree this leaf belongs to and its index within the subtree
    uint subtree_idx = thread_id / uniforms.subtree_leaves_len;
    uint in_subtree_idx = thread_id % uniforms.subtree_leaves_len;

    // Calculate linear digest index
    uint digest_idx = compute_linear_leaf_index(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        in_subtree_idx
    );

    uint input_offset = thread_id * uniforms.leaf_size;
    uint output_offset = digest_idx * 4;

    // hash_or_noop logic: if leaf_size <= 4, don't hash, just copy
    // This matches Plonky2's CPU behavior where small inputs fit directly in a hash
    if (uniforms.leaf_size <= 4) {
        // No hashing needed - leaf fits in a hash output
        // Copy the leaf elements directly to output (pad with zeros if needed)
        for (uint i = 0; i < 4; i++) {
            if (i < uniforms.leaf_size) {
                Fp val = leaf_inputs[input_offset + i];
                output[output_offset + i] = static_cast<ulong>(val);
            } else {
                output[output_offset + i] = 0;
            }
        }
    } else {
        // Leaf is larger than a hash - need to hash with Poseidon
        Fp p2_state[12];

        // Initialize all positions to 0 to match CPU sponge initialization
        for (uint i = 0; i < 12; i++) {
            p2_state[i] = 0;
        }

        uint offset = input_offset;
        const constant ulong2* leaf_u2 = (const constant ulong2*)leaf_inputs;
        bool aligned = ((offset & 1u) == 0);

        if (uniforms.leaf_size == 135) {
            // Specialized path: 16 full chunks of 8, then 7 remaining
            #pragma unroll
            for (uint i = 0; i < 16; i++) {
                if (aligned) {
                    uint o2 = offset >> 1;
                    ulong2 v0 = leaf_u2[o2];
                    ulong2 v1 = leaf_u2[o2 + 1];
                    ulong2 v2 = leaf_u2[o2 + 2];
                    ulong2 v3 = leaf_u2[o2 + 3];

                    p2_state[0] = Fp(v0.x);
                    p2_state[1] = Fp(v0.y);
                    p2_state[2] = Fp(v1.x);
                    p2_state[3] = Fp(v1.y);
                    p2_state[4] = Fp(v2.x);
                    p2_state[5] = Fp(v2.y);
                    p2_state[6] = Fp(v3.x);
                    p2_state[7] = Fp(v3.y);
                } else {
                    const constant packed_ulong2* leaf_p2 =
                        (const constant packed_ulong2*)(leaf_inputs + offset);
                    packed_ulong2 v0 = leaf_p2[0];
                    packed_ulong2 v1 = leaf_p2[1];
                    packed_ulong2 v2 = leaf_p2[2];
                    packed_ulong2 v3 = leaf_p2[3];

                    p2_state[0] = Fp(v0.x);
                    p2_state[1] = Fp(v0.y);
                    p2_state[2] = Fp(v1.x);
                    p2_state[3] = Fp(v1.y);
                    p2_state[4] = Fp(v2.x);
                    p2_state[5] = Fp(v2.y);
                    p2_state[6] = Fp(v3.x);
                    p2_state[7] = Fp(v3.y);
                }
                poseidon_permute(p2_state);
                offset += 8;
            }

            p2_state[0] = leaf_inputs[offset + 0];
            p2_state[1] = leaf_inputs[offset + 1];
            p2_state[2] = leaf_inputs[offset + 2];
            p2_state[3] = leaf_inputs[offset + 3];
            p2_state[4] = leaf_inputs[offset + 4];
            p2_state[5] = leaf_inputs[offset + 5];
            p2_state[6] = leaf_inputs[offset + 6];
            poseidon_permute(p2_state);
        } else {
            uint num_full_rounds = uniforms.leaf_size / 8;

            for (uint i = 0; i < num_full_rounds; i++) {
                if (aligned) {
                    uint o2 = offset >> 1;
                    ulong2 v0 = leaf_u2[o2];
                    ulong2 v1 = leaf_u2[o2 + 1];
                    ulong2 v2 = leaf_u2[o2 + 2];
                    ulong2 v3 = leaf_u2[o2 + 3];

                    p2_state[0] = Fp(v0.x);
                    p2_state[1] = Fp(v0.y);
                    p2_state[2] = Fp(v1.x);
                    p2_state[3] = Fp(v1.y);
                    p2_state[4] = Fp(v2.x);
                    p2_state[5] = Fp(v2.y);
                    p2_state[6] = Fp(v3.x);
                    p2_state[7] = Fp(v3.y);
                } else {
                    const constant packed_ulong2* leaf_p2 =
                        (const constant packed_ulong2*)(leaf_inputs + offset);
                    packed_ulong2 v0 = leaf_p2[0];
                    packed_ulong2 v1 = leaf_p2[1];
                    packed_ulong2 v2 = leaf_p2[2];
                    packed_ulong2 v3 = leaf_p2[3];

                    p2_state[0] = Fp(v0.x);
                    p2_state[1] = Fp(v0.y);
                    p2_state[2] = Fp(v1.x);
                    p2_state[3] = Fp(v1.y);
                    p2_state[4] = Fp(v2.x);
                    p2_state[5] = Fp(v2.y);
                    p2_state[6] = Fp(v3.x);
                    p2_state[7] = Fp(v3.y);
                }
                poseidon_permute(p2_state);
                offset += 8;
            }

            uint remaining = uniforms.leaf_size - num_full_rounds * 8;
            if (remaining != 0) {
                for (uint i = 0; i < remaining; i++) {
                    p2_state[i] = leaf_inputs[offset + i];
                }
                poseidon_permute(p2_state);
            }
        }

        // Write the hashed result
        output[output_offset] = static_cast<ulong>(p2_state[0]);
        output[output_offset + 1] = static_cast<ulong>(p2_state[1]);
        output[output_offset + 2] = static_cast<ulong>(p2_state[2]);
        output[output_offset + 3] = static_cast<ulong>(p2_state[3]);
    }
}

// Kernel to hash internal tree levels with linear layout
// This is called for each level from leaves-1 up to level 1 (just below cap)
kernel void poseidon_hash_tree_level_linear(
    device ulong * output[[buffer(0)]],
    constant LinearUniforms & uniforms[[buffer(1)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];

    // Number of nodes at this level across all subtrees
    uint nodes_per_subtree = uniforms.subtree_leaves_len >> uniforms.level;
    uint total_nodes = nodes_per_subtree * uniforms.subtree_count;

    if (thread_id >= total_nodes) {
        return;
    }

    // Determine which subtree and which node within subtree
    uint subtree_idx = thread_id / nodes_per_subtree;
    uint index_in_level = thread_id % nodes_per_subtree;

    // Calculate parent index (where we write)
    uint parent_idx = compute_linear_internal_index(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        uniforms.level,
        index_in_level
    );

    // Calculate children indices (where we read from)
    // Children are at level (uniforms.level - 1)
    uint child_level = uniforms.level - 1;
    uint left_child_in_level = index_in_level * 2;
    uint right_child_in_level = index_in_level * 2 + 1;

    uint left_child_idx, right_child_idx;

    if (child_level == 0) {
        // Children are leaves
        left_child_idx = compute_linear_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            left_child_in_level
        );
        right_child_idx = compute_linear_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            right_child_in_level
        );
    } else {
        // Children are internal nodes
        left_child_idx = compute_linear_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            left_child_in_level
        );
        right_child_idx = compute_linear_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            right_child_in_level
        );
    }

    // Read children (8 elements total: 4 for left, 4 for right)
    Fp p2_state[12];
    uint left_offset = left_child_idx * 4;
    uint right_offset = right_child_idx * 4;

    device const ulong2* out2 = (device const ulong2*)output;
    uint lo2 = left_offset >> 1;
    uint ro2 = right_offset >> 1;

    ulong2 l0 = out2[lo2];
    ulong2 l1 = out2[lo2 + 1];
    ulong2 r0 = out2[ro2];
    ulong2 r1 = out2[ro2 + 1];

    p2_state[0] = Fp(l0.x);
    p2_state[1] = Fp(l0.y);
    p2_state[2] = Fp(l1.x);
    p2_state[3] = Fp(l1.y);

    p2_state[4] = Fp(r0.x);
    p2_state[5] = Fp(r0.y);
    p2_state[6] = Fp(r1.x);
    p2_state[7] = Fp(r1.y);

    // Zero out remaining state
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;

    // Hash
    poseidon_permute(p2_state);

    // Write parent (4 elements)
    uint parent_offset = parent_idx * 4;
    output[parent_offset] = static_cast<ulong>(p2_state[0]);
    output[parent_offset + 1] = static_cast<ulong>(p2_state[1]);
    output[parent_offset + 2] = static_cast<ulong>(p2_state[2]);
    output[parent_offset + 3] = static_cast<ulong>(p2_state[3]);
}

// Kernel to compute cap hashes from subtree roots
kernel void poseidon_hash_caps_linear(
    device ulong * caps_output[[buffer(0)]],
    device ulong * digests[[buffer(1)]],
    constant LinearUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    if (gid >= uniforms.subtree_count) {
        return;
    }

    // Read the root of this subtree (index 0 within the subtree)
    // For a subtree with root at level tree_height - cap_height - 1,
    // we need to hash its two children which are at level tree_height - cap_height - 2

    // Actually, the cap hash is computed from the two top-level nodes of each subtree
    // The subtree root itself is stored at index 0 of each subtree's digest array
    // But for the cap, we hash from the two children of the cap level

    // In linear layout, subtree root is at index 0 of subtree
    // Its left child is at index 1, right child is at index 2
    uint subtree_base = gid * uniforms.subtree_digests_len;

    // Read children of subtree root (they are at indices 1 and 2)
    uint left_idx = subtree_base + 1;
    uint right_idx = subtree_base + 2;

    Fp p2_state[12];

    p2_state[0] = digests[left_idx * 4];
    p2_state[1] = digests[left_idx * 4 + 1];
    p2_state[2] = digests[left_idx * 4 + 2];
    p2_state[3] = digests[left_idx * 4 + 3];

    p2_state[4] = digests[right_idx * 4];
    p2_state[5] = digests[right_idx * 4 + 1];
    p2_state[6] = digests[right_idx * 4 + 2];
    p2_state[7] = digests[right_idx * 4 + 3];

    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;

    poseidon_permute(p2_state);

    // Write cap hash
    uint cap_offset = gid * 4;
    caps_output[cap_offset] = static_cast<ulong>(p2_state[0]);
    caps_output[cap_offset + 1] = static_cast<ulong>(p2_state[1]);
    caps_output[cap_offset + 2] = static_cast<ulong>(p2_state[2]);
    caps_output[cap_offset + 3] = static_cast<ulong>(p2_state[3]);
}
