// Linear indexing Merkle tree hashing with constant-space Poseidon
// Poseidon constants read directly from Metal's constant address space (hardware-cached).
// Threadgroup memory used only for child hash caching in internal kernel.
//
// Memory Layout (for tree with 4 leaves in a subtree):
// Linear Array:
// Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]
// Value:  H0123 H01   H23   L0    L1    L2    L3
//         (root)(lvl1)(lvl1)(leaf)(leaf)(leaf)(leaf)

#include <metal_stdlib>
#include "goldilocks.metal"
#include "poseidon_goldilocks.metal"
#include "poseidon_goldilocks_threadgroup.metal"

using namespace metal;
using namespace GoldilocksField;

// Uniforms for linear layout with threadgroup support
struct LinearThreadgroupUniforms {
    uint level;               // Current tree level being processed (0 = leaf level)
    uint subtree_digests_len; // Number of digests per subtree (2 * leaves - 1)
    uint subtree_leaves_len;  // Number of leaves per subtree
    uint leaf_size;           // Elements per leaf
    uint leaf_count;          // Total number of leaves
    uint subtree_count;       // Number of subtrees (2^cap_height)
    uint grid_width;          // Width of the dispatch grid (threads in X dimension)
};

// Compute digest index for a leaf in linear layout
inline uint compute_linear_leaf_index_tg(
    uint subtree_digests_len,
    uint subtree_leaves_len,
    uint subtree_idx,
    uint in_subtree_idx
) {
    return subtree_idx * subtree_digests_len
         + (subtree_digests_len - subtree_leaves_len)
         + in_subtree_idx;
}

// Compute digest index for an internal node in linear layout
inline uint compute_linear_internal_index_tg(
    uint subtree_digests_len,
    uint subtree_leaves_len,
    uint subtree_idx,
    uint level,
    uint index_in_level
) {
    uint level_start = subtree_digests_len - 2 * subtree_leaves_len + (subtree_leaves_len >> level);
    return subtree_idx * subtree_digests_len + level_start + index_in_level;
}

// Kernel to hash leaves with linear layout, constants read from constant address space
kernel void poseidon_hash_leaves_linear_threadgroup(
    constant Fp * leaf_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
    constant LinearThreadgroupUniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];

    // Bounds check
    if (thread_id >= uniforms.leaf_count) {
        return;
    }

    // Calculate which subtree this leaf belongs to and its index within the subtree
    uint subtree_idx = thread_id / uniforms.subtree_leaves_len;
    uint in_subtree_idx = thread_id % uniforms.subtree_leaves_len;

    // Calculate linear digest index
    uint digest_idx = compute_linear_leaf_index_tg(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        in_subtree_idx
    );

    uint input_offset = thread_id * uniforms.leaf_size;
    uint output_offset = digest_idx * 4;

    // hash_or_noop logic
    if (uniforms.leaf_size <= 4) {
        for (uint i = 0; i < 4; i++) {
            if (i < uniforms.leaf_size) {
                Fp val = leaf_inputs[input_offset + i];
                output[output_offset + i] = static_cast<ulong>(val);
            } else {
                output[output_offset + i] = 0;
            }
        }
    } else {
        Fp p2_state[12];

        for (uint i = 0; i < 12; i++) {
            p2_state[i] = 0;
        }

        uint offset = input_offset;
        uint num_full_rounds = uniforms.leaf_size / 8;

        for (uint i = 0; i < num_full_rounds; i++) {
            p2_state[0] = leaf_inputs[offset];
            p2_state[1] = leaf_inputs[offset + 1];
            p2_state[2] = leaf_inputs[offset + 2];
            p2_state[3] = leaf_inputs[offset + 3];

            p2_state[4] = leaf_inputs[offset + 4];
            p2_state[5] = leaf_inputs[offset + 5];
            p2_state[6] = leaf_inputs[offset + 6];
            p2_state[7] = leaf_inputs[offset + 7];

            poseidon_permute_const(p2_state);
            offset += 8;
        }

        uint remaining = uniforms.leaf_size - num_full_rounds * 8;
        if (remaining != 0) {
            for (uint i = 0; i < remaining; i++) {
                p2_state[i] = leaf_inputs[offset + i];
            }
            poseidon_permute_const(p2_state);
        }

        output[output_offset] = static_cast<ulong>(p2_state[0]);
        output[output_offset + 1] = static_cast<ulong>(p2_state[1]);
        output[output_offset + 2] = static_cast<ulong>(p2_state[2]);
        output[output_offset + 3] = static_cast<ulong>(p2_state[3]);
    }
}

// Kernel to hash internal tree levels with linear layout
// Uses threadgroup memory for child hash data only; constants read from constant address space
kernel void poseidon_hash_tree_level_linear_threadgroup(
    device ulong * output[[buffer(0)]],
    constant LinearThreadgroupUniforms & uniforms[[buffer(1)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],  // Child cache only (8 ulongs per thread)
    uint2 gid[[thread_position_in_grid]],
    uint2 lid[[thread_position_in_threadgroup]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];
    uint local_id = lid[0];

    // Threadgroup memory used for child cache only
    threadgroup ulong * shared_children = tg_memory;

    // Number of nodes at this level across all subtrees
    uint nodes_per_subtree = uniforms.subtree_leaves_len >> uniforms.level;
    uint total_nodes = nodes_per_subtree * uniforms.subtree_count;

    if (thread_id >= total_nodes) {
        return;
    }

    // Determine which subtree and which node within subtree
    uint subtree_idx = thread_id / nodes_per_subtree;
    uint index_in_level = thread_id % nodes_per_subtree;

    // Calculate children indices
    uint child_level = uniforms.level - 1;
    uint left_child_in_level = index_in_level * 2;
    uint right_child_in_level = index_in_level * 2 + 1;

    uint left_child_idx, right_child_idx;

    if (child_level == 0) {
        // Children are leaves
        left_child_idx = compute_linear_leaf_index_tg(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            left_child_in_level
        );
        right_child_idx = compute_linear_leaf_index_tg(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            right_child_in_level
        );
    } else {
        // Children are internal nodes
        left_child_idx = compute_linear_internal_index_tg(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            left_child_in_level
        );
        right_child_idx = compute_linear_internal_index_tg(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            right_child_in_level
        );
    }

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
    p2_state[2] = Fp(shared_children[shared_base + 2]);
    p2_state[3] = Fp(shared_children[shared_base + 3]);
    p2_state[4] = Fp(shared_children[shared_base + 4]);
    p2_state[5] = Fp(shared_children[shared_base + 5]);
    p2_state[6] = Fp(shared_children[shared_base + 6]);
    p2_state[7] = Fp(shared_children[shared_base + 7]);
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;

    poseidon_permute_const(p2_state);

    // Calculate parent index (where we write)
    uint parent_idx = compute_linear_internal_index_tg(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        uniforms.level,
        index_in_level
    );

    // Write parent
    uint parent_offset = parent_idx * 4;
    output[parent_offset] = static_cast<ulong>(p2_state[0]);
    output[parent_offset + 1] = static_cast<ulong>(p2_state[1]);
    output[parent_offset + 2] = static_cast<ulong>(p2_state[2]);
    output[parent_offset + 3] = static_cast<ulong>(p2_state[3]);
}

// Kernel to compute cap hashes from subtree roots
// Constants read from constant address space
kernel void poseidon_hash_caps_linear_threadgroup(
    device ulong * caps_output[[buffer(0)]],
    device ulong * digests[[buffer(1)]],
    constant LinearThreadgroupUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    if (gid >= uniforms.subtree_count) {
        return;
    }

    uint subtree_base = gid * uniforms.subtree_digests_len;

    // Read children of subtree root (indices 0 and 1 in BFS layout)
    uint left_idx = subtree_base;
    uint right_idx = subtree_base + 1;

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

    poseidon_permute_const(p2_state);

    // Write cap hash
    uint cap_offset = gid * 4;
    caps_output[cap_offset] = static_cast<ulong>(p2_state[0]);
    caps_output[cap_offset + 1] = static_cast<ulong>(p2_state[1]);
    caps_output[cap_offset + 2] = static_cast<ulong>(p2_state[2]);
    caps_output[cap_offset + 3] = static_cast<ulong>(p2_state[3]);
}
