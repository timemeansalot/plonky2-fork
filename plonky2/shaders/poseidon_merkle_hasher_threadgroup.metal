// Threadgroup-optimized Merkle tree hashing
// Uses threadgroup memory for caching:
// 1. Poseidon round constants (360 ulongs = 2880 bytes)
// 2. MDS frequency constants (12 longs = 96 bytes)
// 3. Child hash data for internal node computation
//
// This reduces device memory bandwidth by caching frequently accessed data

#include <metal_stdlib>
#include "goldilocks.metal"
#include "poseidon_goldilocks.metal"
#include "poseidon_goldilocks_threadgroup.metal"

using namespace metal;
using namespace GoldilocksField;

inline uint compute_digest_index_tg(uint tree_length, uint num_layers, uint level_in, uint index_in) {
    ulong index = (ulong)index_in;
    ulong level = (ulong)level_in;

    ulong leaf_index = index<<(level);
    ulong tree_index = index>>(((ulong)num_layers)-level);
    ulong pair_index = ((leaf_index)&((1 << ((ulong)num_layers)) - 1))>>level;
    ulong parity = pair_index&1;
    pair_index >>=1;

    ulong siblings_index = (pair_index << (level + 1)) + (1 << level) - 1;
    ulong d_index = 2 * siblings_index + parity;
    return (uint)(d_index+((ulong)tree_index*(ulong)tree_length));
}

struct ThreadgroupUniforms {
    uint level;
    uint tree_length;
    uint num_layers;
    uint leaf_size;
    uint leaf_count;
    uint grid_width;
};

// Threadgroup-optimized leaf hashing kernel
// Uses threadgroup memory to cache round constants and MDS constants
kernel void poseidon_hash_leaves_threadgroup(
    constant Fp * leaf_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
    constant ThreadgroupUniforms & uniforms[[buffer(2)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],  // RC + MDS constants
    uint2 gid[[thread_position_in_grid]],
    uint2 lid[[thread_position_in_threadgroup]],
    uint2 tg_size[[threads_per_threadgroup]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];
    uint local_id = lid[0];
    uint num_threads_in_group = tg_size[0];

    // Split threadgroup memory: round constants then MDS constants
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = (threadgroup long *)(tg_memory + POSEIDON_RC_TOTAL);

    // Cooperatively load all constants into threadgroup memory
    load_all_constants_tg(local_id, num_threads_in_group, tg_round_constants, tg_mds_constants);

    // Synchronize to ensure all constants are loaded
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bounds check
    if (thread_id >= uniforms.leaf_count) {
        return;
    }

    uint input_offset = thread_id * uniforms.leaf_size;
    uint digest_idx = compute_digest_index_tg(uniforms.tree_length, uniforms.num_layers, 0, thread_id);
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

            // Use full threadgroup-cached constants (RC + MDS)
            poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants);
            offset += 8;
        }

        uint remaining = uniforms.leaf_size - num_full_rounds * 8;
        if (remaining != 0) {
            for (uint i = 0; i < remaining; i++) {
                p2_state[i] = leaf_inputs[offset + i];
            }
            poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants);
        }

        output[output_offset] = static_cast<ulong>(p2_state[0]);
        output[output_offset + 1] = static_cast<ulong>(p2_state[1]);
        output[output_offset + 2] = static_cast<ulong>(p2_state[2]);
        output[output_offset + 3] = static_cast<ulong>(p2_state[3]);
    }
}

// Threadgroup-optimized tree level hashing kernel
// Uses threadgroup memory for round constants, MDS constants, and child hash values
kernel void poseidon_hash_tree_level_threadgroup(
    constant ThreadgroupUniforms & uniforms[[buffer(1)]],
    device ulong * output[[buffer(0)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],  // Combined: RC + MDS + child cache
    uint2 gid[[thread_position_in_grid]],
    uint2 lid[[thread_position_in_threadgroup]],
    uint2 tg_size[[threads_per_threadgroup]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];
    uint local_id = lid[0];
    uint num_threads_in_group = tg_size[0];

    // Split threadgroup memory: RC + MDS constants, then child cache
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = (threadgroup long *)(tg_memory + POSEIDON_RC_TOTAL);
    threadgroup ulong * shared_children = tg_memory + POSEIDON_RC_TOTAL + MDS_CONST_TOTAL;

    // Cooperatively load all constants into threadgroup memory
    load_all_constants_tg(local_id, num_threads_in_group, tg_round_constants, tg_mds_constants);

    // Synchronize to ensure all constants are loaded
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bounds check
    uint nodes_at_this_level = uniforms.leaf_count >> uniforms.level;
    if (thread_id >= nodes_at_this_level) {
        return;
    }

    // Each thread loads its 8 child values into threadgroup memory
    uint child_offset = compute_digest_index_tg(uniforms.tree_length, uniforms.num_layers, uniforms.level - 1, thread_id * 2) * 4;

    // Load children into threadgroup memory (8 ulongs per thread)
    uint shared_base = local_id * 8;
    shared_children[shared_base + 0] = output[child_offset + 0];
    shared_children[shared_base + 1] = output[child_offset + 1];
    shared_children[shared_base + 2] = output[child_offset + 2];
    shared_children[shared_base + 3] = output[child_offset + 3];
    shared_children[shared_base + 4] = output[child_offset + 4];
    shared_children[shared_base + 5] = output[child_offset + 5];
    shared_children[shared_base + 6] = output[child_offset + 6];
    shared_children[shared_base + 7] = output[child_offset + 7];

    // Barrier to ensure all loads complete
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform hash using data from threadgroup memory
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

    // Use full threadgroup-cached constants (RC + MDS)
    poseidon_permute_tg_full(p2_state, tg_round_constants, tg_mds_constants);

    // Write result
    uint parent_offset = compute_digest_index_tg(uniforms.tree_length, uniforms.num_layers, uniforms.level, thread_id) * 4;
    output[parent_offset] = static_cast<ulong>(p2_state[0]);
    output[parent_offset + 1] = static_cast<ulong>(p2_state[1]);
    output[parent_offset + 2] = static_cast<ulong>(p2_state[2]);
    output[parent_offset + 3] = static_cast<ulong>(p2_state[3]);
}
