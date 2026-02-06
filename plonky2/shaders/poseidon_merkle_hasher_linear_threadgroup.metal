// Linear indexing + Threadgroup memory optimization for Merkle tree hashing
// Combines zeknox-style linear layout with full threadgroup constant caching
//
// Memory Layout (for tree with 4 leaves in a subtree):
// Linear Array:
// Index:  [0]   [1]   [2]   [3]   [4]   [5]   [6]
// Value:  H0123 H01   H23   L0    L1    L2    L3
//         (root)(lvl1)(lvl1)(leaf)(leaf)(leaf)(leaf)
//
// Threadgroup memory is used to cache:
// 1. Poseidon round constants (30 * 12 = 360 ulongs = 2880 bytes)
// 2. MDS frequency constants (12 longs = 96 bytes)
// 3. Child hash data for internal node computation (8 ulongs per thread)
// Total: ~3KB constants + child cache (well within 32KB Metal limit)

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

// Kernel to hash leaves with linear layout and full threadgroup-cached constants
kernel void poseidon_hash_leaves_linear_threadgroup(
    constant Fp * leaf_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
    constant LinearThreadgroupUniforms & uniforms[[buffer(2)]],
    constant uint & flags[[buffer(3)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],  // 360 ulongs for RC + 12 longs for MDS
    uint2 gid[[thread_position_in_grid]],
    uint2 lid[[thread_position_in_threadgroup]],
    uint2 tg_size[[threads_per_threadgroup]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];
    uint local_id = lid[0];
    uint num_threads_in_group = tg_size[0];

    const bool use_tg_consts = (flags & 1u) != 0u;
    // Split threadgroup memory: round constants then MDS constants
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = (threadgroup long *)(tg_memory + POSEIDON_RC_TOTAL);

    if (use_tg_consts) {
        load_all_constants_tg(local_id, num_threads_in_group, tg_round_constants, tg_mds_constants);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

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

            if (use_tg_consts) {
                poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants);
            } else {
                poseidon_permute(p2_state);
            }
            offset += 8;
        }

        uint remaining = uniforms.leaf_size - num_full_rounds * 8;
        if (remaining != 0) {
            for (uint i = 0; i < remaining; i++) {
                p2_state[i] = leaf_inputs[offset + i];
            }
            if (use_tg_consts) {
                poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants);
            } else {
                poseidon_permute(p2_state);
            }
        }

        output[output_offset] = static_cast<ulong>(p2_state[0]);
        output[output_offset + 1] = static_cast<ulong>(p2_state[1]);
        output[output_offset + 2] = static_cast<ulong>(p2_state[2]);
        output[output_offset + 3] = static_cast<ulong>(p2_state[3]);
    }
}

// Kernel to hash leaves and first tree level in one pass
// Each thread handles a pair of leaves, writes both leaf digests and their parent.
kernel void poseidon_hash_leaves_and_level1_linear_threadgroup(
    constant Fp * leaf_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
    constant LinearThreadgroupUniforms & uniforms[[buffer(2)]],
    constant uint & flags[[buffer(3)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],  // 360 ulongs for RC + 12 longs for MDS
    uint2 gid[[thread_position_in_grid]],
    uint2 lid[[thread_position_in_threadgroup]],
    uint2 tg_size[[threads_per_threadgroup]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];
    uint local_id = lid[0];
    uint num_threads_in_group = tg_size[0];
    const bool use_tg_consts = (flags & 1u) != 0u;
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = (threadgroup long *)(tg_memory + POSEIDON_RC_TOTAL);

    if (use_tg_consts) {
        load_all_constants_tg(local_id, num_threads_in_group, tg_round_constants, tg_mds_constants);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint total_pairs = uniforms.leaf_count >> 1;
    if (thread_id >= total_pairs) {
        return;
    }

    uint pairs_per_subtree = uniforms.subtree_leaves_len >> 1;
    uint subtree_idx = thread_id / pairs_per_subtree;
    uint pair_in_subtree = thread_id % pairs_per_subtree;

    uint left_in_subtree = pair_in_subtree * 2;
    uint right_in_subtree = left_in_subtree + 1;

    uint left_digest_idx = compute_linear_leaf_index_tg(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        left_in_subtree
    );
    uint right_digest_idx = compute_linear_leaf_index_tg(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        right_in_subtree
    );
    uint parent_idx = compute_linear_internal_index_tg(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        1,
        pair_in_subtree
    );

    uint left_input_offset = (thread_id * 2) * uniforms.leaf_size;
    uint right_input_offset = left_input_offset + uniforms.leaf_size;

    uint left_output_offset = left_digest_idx * 4;
    uint right_output_offset = right_digest_idx * 4;
    uint parent_output_offset = parent_idx * 4;
    device ulong2* out2 = (device ulong2*)output;

    ulong l0 = 0, l1 = 0, l2 = 0, l3 = 0;
    ulong r0 = 0, r1 = 0, r2 = 0, r3 = 0;

    if (uniforms.leaf_size <= 4) {
        for (uint i = 0; i < 4; i++) {
            ulong lv = 0;
            ulong rv = 0;
            if (i < uniforms.leaf_size) {
                Fp lval = leaf_inputs[left_input_offset + i];
                Fp rval = leaf_inputs[right_input_offset + i];
                lv = static_cast<ulong>(lval);
                rv = static_cast<ulong>(rval);
            }
            if (i == 0) {
                out2[left_output_offset >> 1] = ulong2(lv, 0);
                out2[right_output_offset >> 1] = ulong2(rv, 0);
            } else if (i == 1) {
                ulong2 curL = out2[left_output_offset >> 1];
                curL.y = lv;
                out2[left_output_offset >> 1] = curL;
                ulong2 curR = out2[right_output_offset >> 1];
                curR.y = rv;
                out2[right_output_offset >> 1] = curR;
            } else if (i == 2) {
                out2[(left_output_offset >> 1) + 1] = ulong2(lv, 0);
                out2[(right_output_offset >> 1) + 1] = ulong2(rv, 0);
            } else {
                ulong2 curL = out2[(left_output_offset >> 1) + 1];
                curL.y = lv;
                out2[(left_output_offset >> 1) + 1] = curL;
                ulong2 curR = out2[(right_output_offset >> 1) + 1];
                curR.y = rv;
                out2[(right_output_offset >> 1) + 1] = curR;
            }
            if (i == 0) { l0 = lv; r0 = rv; }
            if (i == 1) { l1 = lv; r1 = rv; }
            if (i == 2) { l2 = lv; r2 = rv; }
            if (i == 3) { l3 = lv; r3 = rv; }
        }
    } else {
        Fp p2_state[12];

        for (uint i = 0; i < 12; i++) {
            p2_state[i] = 0;
        }

        uint offset = left_input_offset;
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

            if (use_tg_consts) {
                poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants);
            } else {
                poseidon_permute(p2_state);
            }
            offset += 8;
        }

        uint remaining = uniforms.leaf_size - num_full_rounds * 8;
        if (remaining != 0) {
            for (uint i = 0; i < remaining; i++) {
                p2_state[i] = leaf_inputs[offset + i];
            }
            if (use_tg_consts) {
                poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants);
            } else {
                poseidon_permute(p2_state);
            }
        }

        l0 = static_cast<ulong>(p2_state[0]);
        l1 = static_cast<ulong>(p2_state[1]);
        l2 = static_cast<ulong>(p2_state[2]);
        l3 = static_cast<ulong>(p2_state[3]);

        out2[left_output_offset >> 1] = ulong2(l0, l1);
        out2[(left_output_offset >> 1) + 1] = ulong2(l2, l3);

        for (uint i = 0; i < 12; i++) {
            p2_state[i] = 0;
        }

        offset = right_input_offset;
        for (uint i = 0; i < num_full_rounds; i++) {
            p2_state[0] = leaf_inputs[offset];
            p2_state[1] = leaf_inputs[offset + 1];
            p2_state[2] = leaf_inputs[offset + 2];
            p2_state[3] = leaf_inputs[offset + 3];

            p2_state[4] = leaf_inputs[offset + 4];
            p2_state[5] = leaf_inputs[offset + 5];
            p2_state[6] = leaf_inputs[offset + 6];
            p2_state[7] = leaf_inputs[offset + 7];

            if (use_tg_consts) {
                poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants);
            } else {
                poseidon_permute(p2_state);
            }
            offset += 8;
        }

        remaining = uniforms.leaf_size - num_full_rounds * 8;
        if (remaining != 0) {
            for (uint i = 0; i < remaining; i++) {
                p2_state[i] = leaf_inputs[offset + i];
            }
            if (use_tg_consts) {
                poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants);
            } else {
                poseidon_permute(p2_state);
            }
        }

        r0 = static_cast<ulong>(p2_state[0]);
        r1 = static_cast<ulong>(p2_state[1]);
        r2 = static_cast<ulong>(p2_state[2]);
        r3 = static_cast<ulong>(p2_state[3]);

        out2[right_output_offset >> 1] = ulong2(r0, r1);
        out2[(right_output_offset >> 1) + 1] = ulong2(r2, r3);
    }

    Fp parent_state[12];
    parent_state[0] = Fp(l0);
    parent_state[1] = Fp(l1);
    parent_state[2] = Fp(l2);
    parent_state[3] = Fp(l3);
    parent_state[4] = Fp(r0);
    parent_state[5] = Fp(r1);
    parent_state[6] = Fp(r2);
    parent_state[7] = Fp(r3);
    parent_state[8] = 0;
    parent_state[9] = 0;
    parent_state[10] = 0;
    parent_state[11] = 0;

    if (use_tg_consts) {
        poseidon_permute_tg_full(parent_state, tg_round_constants, tg_mds_constants);
    } else {
        poseidon_permute(parent_state);
    }

    out2[parent_output_offset >> 1] = ulong2(static_cast<ulong>(parent_state[0]), static_cast<ulong>(parent_state[1]));
    out2[(parent_output_offset >> 1) + 1] = ulong2(static_cast<ulong>(parent_state[2]), static_cast<ulong>(parent_state[3]));
}

// Kernel to hash internal tree levels with linear layout and full threadgroup caching
// Uses threadgroup memory for round constants, MDS constants, and child hash data
kernel void poseidon_hash_tree_level_linear_threadgroup(
    device ulong * output[[buffer(0)]],
    constant LinearThreadgroupUniforms & uniforms[[buffer(1)]],
    constant uint & flags[[buffer(2)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],  // Combined: RC + MDS + child cache
    uint2 gid[[thread_position_in_grid]],
    uint2 lid[[thread_position_in_threadgroup]],
    uint2 tg_size[[threads_per_threadgroup]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];
    uint local_id = lid[0];
    uint num_threads_in_group = tg_size[0];
    (void)local_id;
    (void)num_threads_in_group;

    const bool use_tg_consts = (flags & 1u) != 0u;
    const bool use_child_cache = (flags & 2u) != 0u;
    // Split threadgroup memory: round constants then MDS constants
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = (threadgroup long *)(tg_memory + POSEIDON_RC_TOTAL);

    if (use_tg_consts) {
        load_all_constants_tg(local_id, num_threads_in_group, tg_round_constants, tg_mds_constants);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

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

    // Load children either via shared cache or directly from global memory
    uint left_offset = left_child_idx * 4;
    uint right_offset = right_child_idx * 4;
    ulong2 l0;
    ulong2 l1;
    ulong2 r0;
    ulong2 r1;
    if (use_child_cache) {
        threadgroup ulong * shared_children = tg_memory + (use_tg_consts ? POSEIDON_RC_TOTAL + MDS_CONST_TOTAL : 0);
        uint shared_base = local_id * 8;
        device const ulong2* out2 = (device const ulong2*)output;
        uint lo2 = left_offset >> 1;
        uint ro2 = right_offset >> 1;
        ulong2 tl0 = out2[lo2];
        ulong2 tl1 = out2[lo2 + 1];
        ulong2 tr0 = out2[ro2];
        ulong2 tr1 = out2[ro2 + 1];
        shared_children[shared_base + 0] = tl0.x;
        shared_children[shared_base + 1] = tl0.y;
        shared_children[shared_base + 2] = tl1.x;
        shared_children[shared_base + 3] = tl1.y;
        shared_children[shared_base + 4] = tr0.x;
        shared_children[shared_base + 5] = tr0.y;
        shared_children[shared_base + 6] = tr1.x;
        shared_children[shared_base + 7] = tr1.y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        l0 = ulong2(shared_children[shared_base + 0], shared_children[shared_base + 1]);
        l1 = ulong2(shared_children[shared_base + 2], shared_children[shared_base + 3]);
        r0 = ulong2(shared_children[shared_base + 4], shared_children[shared_base + 5]);
        r1 = ulong2(shared_children[shared_base + 6], shared_children[shared_base + 7]);
    } else {
        device const ulong2* out2 = (device const ulong2*)output;
        uint lo2 = left_offset >> 1;
        uint ro2 = right_offset >> 1;
        l0 = out2[lo2];
        l1 = out2[lo2 + 1];
        r0 = out2[ro2];
        r1 = out2[ro2 + 1];
    }

    // Perform hash using data from threadgroup memory and cached constants
    Fp p2_state[12];
    p2_state[0] = Fp(l0.x);
    p2_state[1] = Fp(l0.y);
    p2_state[2] = Fp(l1.x);
    p2_state[3] = Fp(l1.y);
    p2_state[4] = Fp(r0.x);
    p2_state[5] = Fp(r0.y);
    p2_state[6] = Fp(r1.x);
    p2_state[7] = Fp(r1.y);
    p2_state[8] = 0;
    p2_state[9] = 0;
    p2_state[10] = 0;
    p2_state[11] = 0;

    if (use_tg_consts) {
        poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants);
    } else {
        poseidon_permute(p2_state);
    }

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
    device ulong2* out2w = (device ulong2*)output;
    out2w[parent_offset >> 1] = ulong2(static_cast<ulong>(p2_state[0]), static_cast<ulong>(p2_state[1]));
    out2w[(parent_offset >> 1) + 1] = ulong2(static_cast<ulong>(p2_state[2]), static_cast<ulong>(p2_state[3]));

}

// Kernel to compute cap hashes from subtree roots
// Uses full threadgroup-cached constants (RC + MDS)
kernel void poseidon_hash_caps_linear_threadgroup(
    device ulong * caps_output[[buffer(0)]],
    device ulong * digests[[buffer(1)]],
    constant LinearThreadgroupUniforms & uniforms[[buffer(2)]],
    constant uint & flags[[buffer(3)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],  // RC + MDS constants
    uint gid[[thread_position_in_grid]],
    uint lid[[thread_position_in_threadgroup]],
    uint tg_size[[threads_per_threadgroup]]
) {
    const bool use_tg_consts = (flags & 1u) != 0u;
    // Split threadgroup memory: round constants then MDS constants
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = (threadgroup long *)(tg_memory + POSEIDON_RC_TOTAL);

    if (use_tg_consts) {
        load_all_constants_tg(lid, tg_size, tg_round_constants, tg_mds_constants);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (gid >= uniforms.subtree_count) {
        return;
    }

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

    if (use_tg_consts) {
        poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants);
    } else {
        poseidon_permute(p2_state);
    }

    // Write cap hash
    uint cap_offset = gid * 4;
    caps_output[cap_offset] = static_cast<ulong>(p2_state[0]);
    caps_output[cap_offset + 1] = static_cast<ulong>(p2_state[1]);
    caps_output[cap_offset + 2] = static_cast<ulong>(p2_state[2]);
    caps_output[cap_offset + 3] = static_cast<ulong>(p2_state[3]);
}
