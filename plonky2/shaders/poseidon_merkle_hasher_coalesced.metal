// Coalesced memory access Merkle tree implementation
// Optimized for bandwidth-bound scenarios at large degrees (>= 2^21)
//
// Key optimization: Threads in a threadgroup cooperatively load consecutive memory
// addresses, then access data from shared memory. This maximizes memory bandwidth
// utilization by ensuring coalesced global memory accesses.
//
// Memory Layout (linear, same as poseidon_merkle_hasher_linear.metal):
// For subtree at index subtree_idx with subtree_digests_len total digests:
// - Leaves are stored in the second half: offset = subtree_digests_len - subtree_leaves_len
// - Internal nodes are stored in the first half, level by level from root to leaves-1

#include <metal_stdlib>
#include "goldilocks.metal"
#include "poseidon_goldilocks.metal"
#include "poseidon_goldilocks_threadgroup.metal"

using namespace metal;
using namespace GoldilocksField;

// Uniforms for coalesced layout (extends linear layout with chunk info)
// Uses 2D dispatch: X = node within subtree, Y = subtree index
// This guarantees all threads in a threadgroup are in the same subtree
struct CoalescedUniforms {
    uint level;               // Current tree level being processed (0 = leaf level)
    uint subtree_digests_len; // Number of digests per subtree (2 * leaves - 1)
    uint subtree_leaves_len;  // Number of leaves per subtree
    uint leaf_size;           // Elements per leaf
    uint leaf_count;          // Total number of leaves
    uint subtree_count;       // Number of subtrees (2^cap_height)
    uint nodes_per_subtree;   // Nodes per subtree at this level (for 2D dispatch)
    uint nodes_per_dispatch;  // Number of nodes to process in this dispatch
    uint dispatch_offset;     // Starting node offset for this dispatch
    uint enable_counters;     // Non-zero to enable sequential access counters
};

// Sequential access counters for instrumentation
// When enabled, tracks memory access patterns to verify >80% sequential
struct AccessCounters {
    atomic_uint total_accesses;      // Total number of memory accesses
    atomic_uint sequential_accesses; // Number of sequential (coalesced) accesses
};

// Compute digest index for a leaf in linear layout
inline uint compute_coalesced_leaf_index(
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
inline uint compute_coalesced_internal_index(
    uint subtree_digests_len,
    uint subtree_leaves_len,
    uint subtree_idx,
    uint level,
    uint index_in_level
) {
    uint level_start = subtree_digests_len - 2 * subtree_leaves_len + (subtree_leaves_len >> level);
    return subtree_idx * subtree_digests_len + level_start + index_in_level;
}

// Kernel to hash leaves with coalesced memory access
// Uses cooperative loading where threads in a threadgroup load consecutive leaf data
//
// Chunking Strategy:
// - dispatch_offset specifies the first subtree index for this chunk
// - gid[1] is relative to the chunk, so actual subtree = dispatch_offset + gid[1]
// - This allows processing subsets of subtrees for better cache locality
kernel void poseidon_hash_leaves_coalesced(
    constant Fp * leaf_inputs[[buffer(0)]],
    device ulong * output[[buffer(1)]],
    constant CoalescedUniforms & uniforms[[buffer(2)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],
    uint2 gid[[thread_position_in_grid]],
    uint2 lid[[thread_position_in_threadgroup]],
    uint2 tg_size[[threads_per_threadgroup]]
) {
    // 2D dispatch: gid[0] = node within subtree, gid[1] = relative subtree index in chunk
    // dispatch_offset is the first subtree index in this chunk
    uint in_subtree_idx = gid[0];
    uint subtree_idx = uniforms.dispatch_offset + gid[1];
    uint local_id = lid[0];
    uint num_threads_in_group = tg_size[0];

    // Split threadgroup memory: round constants, MDS constants, then fast partial constants
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = (threadgroup long *)(tg_memory + POSEIDON_RC_TOTAL);
    threadgroup ulong * tg_fast_partial = tg_memory + POSEIDON_RC_TOTAL + MDS_CONST_TOTAL;

    // Cooperatively load all constants into threadgroup memory
    load_all_constants_tg(local_id, num_threads_in_group, tg_round_constants, tg_mds_constants);
    load_fast_partial_constants_tg(local_id, num_threads_in_group, tg_fast_partial);

    // Synchronize to ensure all constants are loaded
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global leaf index using 2D coordinates
    uint leaf_idx = subtree_idx * uniforms.subtree_leaves_len + in_subtree_idx;

    // Bounds check: ensure within subtree and total leaves
    if (in_subtree_idx >= uniforms.subtree_leaves_len || subtree_idx >= uniforms.subtree_count) {
        return;
    }
    if (leaf_idx >= uniforms.leaf_count) {
        return;
    }

    // Calculate linear digest index
    uint digest_idx = compute_coalesced_leaf_index(
        uniforms.subtree_digests_len,
        uniforms.subtree_leaves_len,
        subtree_idx,
        in_subtree_idx
    );

    uint input_offset = leaf_idx * uniforms.leaf_size;
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

            poseidon_permute_tg_fast_partial(p2_state, tg_round_constants, tg_mds_constants, tg_fast_partial);
            offset += 8;
        }

        uint remaining = uniforms.leaf_size - num_full_rounds * 8;
        if (remaining != 0) {
            for (uint i = 0; i < remaining; i++) {
                p2_state[i] = leaf_inputs[offset + i];
            }
            poseidon_permute_tg_fast_partial(p2_state, tg_round_constants, tg_mds_constants, tg_fast_partial);
        }

        output[output_offset] = static_cast<ulong>(p2_state[0]);
        output[output_offset + 1] = static_cast<ulong>(p2_state[1]);
        output[output_offset + 2] = static_cast<ulong>(p2_state[2]);
        output[output_offset + 3] = static_cast<ulong>(p2_state[3]);
    }
}

// Kernel to hash internal tree levels with coalesced memory access
// Key optimization: Threads cooperatively load a contiguous block of children data
// before each thread computes its hash.
//
// For a threadgroup of N threads processing nodes [base, base+N):
// - Children indices for node i are at level (level-1): left=2*i, right=2*i+1
// - For consecutive parent nodes, children form consecutive pairs
// - We load 2*N child hashes (each 4 ulongs) into shared memory
// - Then each thread reads its two children from shared memory
//
// Chunking Strategy:
// - dispatch_offset specifies the first subtree index for this chunk
// - gid[1] is relative to the chunk, so actual subtree = dispatch_offset + gid[1]
// - This allows processing subsets of subtrees for better cache locality
kernel void poseidon_hash_tree_level_coalesced(
    device ulong * output[[buffer(0)]],
    constant CoalescedUniforms & uniforms[[buffer(1)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],
    uint2 gid[[thread_position_in_grid]],
    uint2 lid[[thread_position_in_threadgroup]],
    uint2 tg_size[[threads_per_threadgroup]]
) {
    // 2D dispatch: gid[0] = node within subtree (may be chunked), gid[1] = subtree index
    // dispatch_offset is X-dimension offset for sublevel chunking (AC-1)
    // The actual index_in_level = dispatch_offset + gid[0]
    uint index_in_level = uniforms.dispatch_offset + gid[0];
    uint subtree_idx = gid[1];
    uint local_id = lid[0];
    uint num_threads_in_group = tg_size[0];

    // Split threadgroup memory:
    // [0..359] = round constants
    // [360..371] = MDS constants
    // [372..1010] = fast partial constants
    // [1011+] = child hash cache (8 ulongs per thread = 2 children * 4 elements)
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = (threadgroup long *)(tg_memory + POSEIDON_RC_TOTAL);
    threadgroup ulong * tg_fast_partial = tg_memory + POSEIDON_RC_TOTAL + MDS_CONST_TOTAL;
    threadgroup ulong * shared_children = tg_memory + POSEIDON_RC_TOTAL + MDS_CONST_TOTAL + FAST_PARTIAL_CONST_TOTAL;

    // Cooperatively load all constants into threadgroup memory
    load_all_constants_tg(local_id, num_threads_in_group, tg_round_constants, tg_mds_constants);
    load_fast_partial_constants_tg(local_id, num_threads_in_group, tg_fast_partial);

    // Synchronize to ensure all constants are loaded
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Number of nodes at this level per subtree
    uint nodes_per_subtree = uniforms.nodes_per_subtree;

    // Check if this thread is active (within bounds)
    // Note: We don't early-return here because ALL threads must participate in cooperative load
    bool is_active = (index_in_level < nodes_per_subtree && subtree_idx < uniforms.subtree_count);

    // Calculate children indices (only needed for active threads, but compute anyway to avoid divergence)
    uint child_level = uniforms.level - 1;
    uint left_child_in_level = index_in_level * 2;
    uint right_child_in_level = index_in_level * 2 + 1;

    uint left_child_idx, right_child_idx;

    if (child_level == 0) {
        // Children are leaves
        left_child_idx = compute_coalesced_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            left_child_in_level
        );
        right_child_idx = compute_coalesced_leaf_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            right_child_in_level
        );
    } else {
        // Children are internal nodes
        left_child_idx = compute_coalesced_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            left_child_in_level
        );
        right_child_idx = compute_coalesced_internal_index(
            uniforms.subtree_digests_len,
            uniforms.subtree_leaves_len,
            subtree_idx,
            child_level,
            right_child_in_level
        );
    }

    // ============ Guaranteed Cooperative Coalesced Load Pattern ============
    // With 2D dispatch (X=node in subtree, Y=subtree), ALL threads in a threadgroup
    // are GUARANTEED to be in the same subtree. No voting or fallback needed.
    //
    // IMPORTANT: For the LAST threadgroup in each subtree, there may be fewer
    // remaining nodes than threadgroup_size. We must compute the actual number
    // of nodes this threadgroup should process.

    // Calculate the base index (first node index in this threadgroup)
    // index_in_level = gid[0], local_id is thread's position within threadgroup
    // So group_base = index_in_level - local_id gives us the first node index
    uint group_base = index_in_level - local_id;

    // Calculate how many nodes remain from group_base to end of subtree
    uint remaining_nodes = (nodes_per_subtree > group_base) ? (nodes_per_subtree - group_base) : 0;

    // Calculate base offset for thread 0's left child in this threadgroup
    // Initialize to 0 to silence compiler warning; thread 0 overwrites before barrier
    threadgroup uint tg_base_child_offset = 0;
    threadgroup uint tg_effective_threads = 0;
    if (local_id == 0) {
        tg_base_child_offset = left_child_idx * 4;
        // Effective threads = min(threadgroup_size, remaining_nodes)
        // This handles the last threadgroup which may have fewer nodes
        tg_effective_threads = min(num_threads_in_group, remaining_nodes);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint shared_base = local_id * 8;

    // ============ COOPERATIVE COALESCED PATH (GUARANTEED) ============
    // All threads in threadgroup are in the same subtree (by 2D dispatch design)
    // Load only what's needed for active threads: 8 ulongs per active thread
    uint total_elements = tg_effective_threads * 8;

    // Cooperative load: thread i loads elements i, i+N, i+2N, ...
    // This gives perfect coalescing: consecutive threads access consecutive addresses
    // All threads participate in the load loop (even inactive ones help load data)
    for (uint i = local_id; i < total_elements; i += num_threads_in_group) {
        shared_children[i] = output[tg_base_child_offset + i];
    }

    // Barrier to ensure all cooperative loads complete
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now early-return for inactive threads (after cooperative load is done)
    if (!is_active) {
        return;
    }

    // Compute from shared memory (fast L1-like access)
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

    // Use fast partial round optimization (RC + MDS + fast partial)
    poseidon_permute_tg_fast_partial(p2_state, tg_round_constants, tg_mds_constants, tg_fast_partial);

    // Calculate parent index (where we write)
    uint parent_idx = compute_coalesced_internal_index(
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

// Instrumented kernel for measuring sequential access patterns
// Same as poseidon_hash_tree_level_coalesced but with counter collection
// Use this kernel when uniforms.enable_counters is non-zero
//
// Chunking Strategy (same as main kernel):
// - dispatch_offset specifies the first subtree index for this chunk
// - dispatch_offset is X-dimension offset for sublevel chunking (AC-1)
kernel void poseidon_hash_tree_level_coalesced_instrumented(
    device ulong * output[[buffer(0)]],
    constant CoalescedUniforms & uniforms[[buffer(1)]],
    device AccessCounters * counters[[buffer(2)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],
    uint2 gid[[thread_position_in_grid]],
    uint2 lid[[thread_position_in_threadgroup]],
    uint2 tg_size[[threads_per_threadgroup]]
) {
    // 2D dispatch: gid[0] = node within subtree (may be chunked), gid[1] = subtree index
    // dispatch_offset is X-dimension offset for sublevel chunking (AC-1)
    uint index_in_level = uniforms.dispatch_offset + gid[0];
    uint subtree_idx = gid[1];
    uint local_id = lid[0];
    uint num_threads_in_group = tg_size[0];

    // Split threadgroup memory
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = (threadgroup long *)(tg_memory + POSEIDON_RC_TOTAL);
    threadgroup ulong * tg_fast_partial = tg_memory + POSEIDON_RC_TOTAL + MDS_CONST_TOTAL;
    threadgroup ulong * shared_children = tg_memory + POSEIDON_RC_TOTAL + MDS_CONST_TOTAL + FAST_PARTIAL_CONST_TOTAL;
    // Extra space for tracking addresses
    threadgroup uint * tg_addresses = (threadgroup uint *)(shared_children + num_threads_in_group * 8);

    load_all_constants_tg(local_id, num_threads_in_group, tg_round_constants, tg_mds_constants);
    load_fast_partial_constants_tg(local_id, num_threads_in_group, tg_fast_partial);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nodes_per_subtree = uniforms.nodes_per_subtree;
    bool is_active = (index_in_level < nodes_per_subtree && subtree_idx < uniforms.subtree_count);

    uint child_level = uniforms.level - 1;
    uint left_child_in_level = index_in_level * 2;

    uint left_child_idx;
    if (child_level == 0) {
        left_child_idx = compute_coalesced_leaf_index(
            uniforms.subtree_digests_len, uniforms.subtree_leaves_len,
            subtree_idx, left_child_in_level);
    } else {
        left_child_idx = compute_coalesced_internal_index(
            uniforms.subtree_digests_len, uniforms.subtree_leaves_len,
            subtree_idx, child_level, left_child_in_level);
    }

    uint group_base = index_in_level - local_id;
    uint remaining_nodes = (nodes_per_subtree > group_base) ? (nodes_per_subtree - group_base) : 0;

    threadgroup uint tg_base_child_offset = 0;
    threadgroup uint tg_effective_threads = 0;
    if (local_id == 0) {
        tg_base_child_offset = left_child_idx * 4;
        tg_effective_threads = min(num_threads_in_group, remaining_nodes);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store each thread's first load address for sequential analysis
    uint my_first_addr = tg_base_child_offset + local_id;
    if (local_id < tg_effective_threads) {
        tg_addresses[local_id] = my_first_addr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 counts sequential accesses within this threadgroup
    if (local_id == 0 && tg_effective_threads > 0) {
        uint total = tg_effective_threads;
        uint sequential = 0;
        for (uint i = 1; i < tg_effective_threads; i++) {
            // Sequential if addresses differ by exactly 1 (consecutive threads load consecutive addresses)
            if (tg_addresses[i] == tg_addresses[i-1] + 1) {
                sequential++;
            }
        }
        // All accesses are coalesced if consecutive threads have consecutive addresses
        // +1 because we count pairs, so N threads = N-1 pairs, and if all sequential, that's N-1 sequential out of N-1 total pairs
        // We record total_accesses as effective_threads and sequential_accesses as sequential pairs + 1 (first access is always "sequential" with itself)
        atomic_fetch_add_explicit(&counters->total_accesses, total, memory_order_relaxed);
        atomic_fetch_add_explicit(&counters->sequential_accesses, sequential + 1, memory_order_relaxed);
    }

    // Cooperative load (same as non-instrumented version)
    uint total_elements = tg_effective_threads * 8;
    for (uint i = local_id; i < total_elements; i += num_threads_in_group) {
        shared_children[i] = output[tg_base_child_offset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!is_active) {
        return;
    }

    // Compute from shared memory
    uint shared_base = local_id * 8;
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

    poseidon_permute_tg_fast_partial(p2_state, tg_round_constants, tg_mds_constants, tg_fast_partial);

    uint parent_idx = compute_coalesced_internal_index(
        uniforms.subtree_digests_len, uniforms.subtree_leaves_len,
        subtree_idx, uniforms.level, index_in_level);

    uint parent_offset = parent_idx * 4;
    output[parent_offset] = static_cast<ulong>(p2_state[0]);
    output[parent_offset + 1] = static_cast<ulong>(p2_state[1]);
    output[parent_offset + 2] = static_cast<ulong>(p2_state[2]);
    output[parent_offset + 3] = static_cast<ulong>(p2_state[3]);
}

// Kernel to compute cap hashes from subtree roots
kernel void poseidon_hash_caps_coalesced(
    device ulong * caps_output[[buffer(0)]],
    device ulong * digests[[buffer(1)]],
    constant CoalescedUniforms & uniforms[[buffer(2)]],
    threadgroup ulong * tg_memory[[threadgroup(0)]],
    uint gid[[thread_position_in_grid]],
    uint lid[[thread_position_in_threadgroup]],
    uint tg_size[[threads_per_threadgroup]]
) {
    // Split threadgroup memory: round constants, MDS constants, then fast partial constants
    threadgroup ulong * tg_round_constants = tg_memory;
    threadgroup long * tg_mds_constants = (threadgroup long *)(tg_memory + POSEIDON_RC_TOTAL);
    threadgroup ulong * tg_fast_partial = tg_memory + POSEIDON_RC_TOTAL + MDS_CONST_TOTAL;

    // Cooperatively load all constants into threadgroup memory
    load_all_constants_tg(lid, tg_size, tg_round_constants, tg_mds_constants);
    load_fast_partial_constants_tg(lid, tg_size, tg_fast_partial);

    // Synchronize to ensure all constants are loaded
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    // Use fast partial round optimization (RC + MDS + fast partial)
    poseidon_permute_tg_fast_partial(p2_state, tg_round_constants, tg_mds_constants, tg_fast_partial);

    // Write cap hash
    uint cap_offset = gid * 4;
    caps_output[cap_offset] = static_cast<ulong>(p2_state[0]);
    caps_output[cap_offset + 1] = static_cast<ulong>(p2_state[1]);
    caps_output[cap_offset + 2] = static_cast<ulong>(p2_state[2]);
    caps_output[cap_offset + 3] = static_cast<ulong>(p2_state[3]);
}
