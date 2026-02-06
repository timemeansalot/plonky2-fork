#include <metal_stdlib>

#include "goldilocks.metal"

#include "poseidon_goldilocks.metal"

using namespace metal;
using namespace GoldilocksField;


inline uint compute_digest_index(uint tree_length, uint num_layers, uint level_in, uint index_in) {
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

struct Uniforms {
    uint level;
    uint tree_length;
    uint num_layers;
    uint leaf_size;
    uint leaf_count;
    uint grid_width;  // Width of the dispatch grid (threads in X dimension)
};

kernel void poseidon_hash_leaves(
    constant Fp * leaf_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
    constant Uniforms & uniforms[[buffer(2)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];

    // Bounds check: don't process threads beyond the actual leaf count
    if (thread_id >= uniforms.leaf_count) {
        return;
    }

    uint input_offset = thread_id * uniforms.leaf_size;
    uint digest_idx = compute_digest_index(uniforms.tree_length, uniforms.num_layers, 0, thread_id);
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

kernel void poseidon_hash_tree_level(
    constant Uniforms & uniforms[[buffer(1)]],
    device ulong * output[[buffer(0)]],
    uint2 gid[[thread_position_in_grid]]
) {
    uint thread_id = gid[1] * uniforms.grid_width + gid[0];

    // Bounds check: at tree level i, there are (leaf_count >> i) nodes to process
    // Each thread processes one node
    uint nodes_at_this_level = uniforms.leaf_count >> uniforms.level;
    if (thread_id >= nodes_at_this_level) {
        return;
    }

    Fp p2_state[12];
    uint offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, uniforms.level - 1, thread_id * 2) * 4;

    device const ulong2* out2 = (device const ulong2*)output;
    uint o2 = offset >> 1;
    ulong2 l0 = out2[o2];
    ulong2 l1 = out2[o2 + 1];
    ulong2 r0 = out2[o2 + 2];
    ulong2 r1 = out2[o2 + 3];

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
    poseidon_permute(p2_state);

    offset = compute_digest_index(uniforms.tree_length, uniforms.num_layers, uniforms.level, thread_id) * 4;

    output[offset] = static_cast<ulong>(p2_state[0]);
    output[offset + 1] = static_cast<ulong>(p2_state[1]);
    output[offset + 2] = static_cast<ulong>(p2_state[2]);
    output[offset + 3] = static_cast<ulong>(p2_state[3]);
}
