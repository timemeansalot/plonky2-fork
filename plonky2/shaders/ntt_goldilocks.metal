// NTT (Number Theoretic Transform) implementation for Goldilocks field
// Based on zeknox CUDA implementation patterns

#include <metal_stdlib>
#include "goldilocks.metal"

using namespace metal;
using namespace GoldilocksField;

// Goldilocks prime: p = 2^64 - 2^32 + 1
constant ulong GL_PRIME = 18446744069414584321UL;

// Helper: Goldilocks addition with reduction
inline ulong gl_add(ulong a, ulong b) {
    ulong sum = a + b;
    // If overflow occurred or sum >= prime, reduce
    if (sum < a || sum >= GL_PRIME) {
        sum -= GL_PRIME;
    }
    return sum;
}

// Helper: Goldilocks subtraction with reduction
inline ulong gl_sub(ulong a, ulong b) {
    if (a >= b) {
        return a - b;
    } else {
        // a < b, so result would be negative
        // return a - b + p = a + (p - b)
        return a + (GL_PRIME - b);
    }
}

// Helper: Goldilocks multiplication using 128-bit intermediate
// Uses the special structure of Goldilocks prime for fast reduction
inline ulong gl_mul(ulong a, ulong b) {
    // Compute a * b as 128-bit value using 32-bit word splitting
    ulong a_lo = a & 0xFFFFFFFF;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFF;
    ulong b_hi = b >> 32;

    // Four 64-bit products
    ulong p_ll = a_lo * b_lo;
    ulong p_lh = a_lo * b_hi;
    ulong p_hl = a_hi * b_lo;
    ulong p_hh = a_hi * b_hi;

    // Combine into 128-bit result
    // result = p_hh * 2^64 + (p_lh + p_hl) * 2^32 + p_ll
    ulong mid = p_lh + p_hl;
    ulong mid_carry = (mid < p_lh) ? 1UL : 0UL;

    ulong lo = p_ll + (mid << 32);
    ulong lo_carry = (lo < p_ll) ? 1UL : 0UL;

    ulong hi = p_hh + (mid >> 32) + (mid_carry << 32) + lo_carry;

    // Reduce 128-bit value modulo Goldilocks prime
    // For Goldilocks: x mod p = x_lo - x_hi * (2^32 - 1) mod p
    // Since 2^64 ≡ 2^32 - 1 (mod p)

    // Use reduce128 from goldilocks.metal
    return reduce128(hi, lo) % GL_PRIME;
}

// NTT Uniforms
struct NTTUniforms {
    uint n;           // Transform size (power of 2)
    uint log_n;       // log2(n)
    uint stage;       // Current butterfly stage
    uint direction;   // 0 = forward NTT, 1 = inverse NTT
    uint twiddle_stride; // Stride in twiddle table for this NTT size
};

// Bit reversal kernel
// Reorders elements according to bit-reversed indices
kernel void ntt_bit_reverse(
    device ulong * data[[buffer(0)]],
    constant NTTUniforms & uniforms[[buffer(1)]],
    uint gid[[thread_position_in_grid]]
) {
    if (gid >= uniforms.n) return;

    // Compute bit-reversed index
    uint rev = 0;
    uint x = gid;
    for (uint i = 0; i < uniforms.log_n; i++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }

    // Only swap if gid < rev (to avoid double-swapping)
    if (gid < rev) {
        ulong temp = data[gid];
        data[gid] = data[rev];
        data[rev] = temp;
    }
}

// Butterfly kernel for NTT
// Performs one stage of Cooley-Tukey butterfly operations
kernel void ntt_butterfly(
    device ulong * data[[buffer(0)]],
    constant ulong * twiddles[[buffer(1)]],
    constant NTTUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    uint num_butterflies = uniforms.n / 2;
    if (gid >= num_butterflies) return;

    uint stage = uniforms.stage;
    uint stride = 1u << stage;
    uint m = 1u << (stage + 1);

    // Calculate butterfly pair indices
    uint k = gid % stride;
    uint j = 2 * stride * (gid / stride) + k;
    uint i = j + stride;

    // Twiddle factor index: k * (n / m) gives the power of omega needed
    // Multiply by twiddle_stride to get index in the precomputed table
    uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
    ulong w = twiddles[twiddle_idx];

    // Butterfly operation
    ulong u = data[j];
    ulong v = gl_mul(data[i], w);

    data[j] = gl_add(u, v);
    data[i] = gl_sub(u, v);
}

// INTT (Inverse NTT) butterfly kernel
// Same as NTT but uses inverse twiddle factors
kernel void intt_butterfly(
    device ulong * data[[buffer(0)]],
    constant ulong * inv_twiddles[[buffer(1)]],
    constant NTTUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    uint num_butterflies = uniforms.n / 2;
    if (gid >= num_butterflies) return;

    uint stage = uniforms.stage;
    uint stride = 1u << stage;
    uint m = 1u << (stage + 1);

    // Calculate butterfly pair indices
    uint k = gid % stride;
    uint j = 2 * stride * (gid / stride) + k;
    uint i = j + stride;

    // Inverse twiddle factor index with stride
    uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
    ulong w = inv_twiddles[twiddle_idx];

    // Butterfly operation (same as forward)
    ulong u = data[j];
    ulong v = gl_mul(data[i], w);

    data[j] = gl_add(u, v);
    data[i] = gl_sub(u, v);
}

// Scale kernel for INTT normalization
// Multiplies all elements by n^(-1) mod p
kernel void ntt_scale(
    device ulong * data[[buffer(0)]],
    constant ulong & n_inv[[buffer(1)]],
    constant uint & n[[buffer(2)]],
    uint gid[[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = gl_mul(data[gid], n_inv);
}

// Coalesced NTT kernel - processes multiple stages with better memory access
// Each thread handles multiple butterfly operations
kernel void ntt_butterfly_coalesced(
    device ulong * data[[buffer(0)]],
    constant ulong * twiddles[[buffer(1)]],
    constant NTTUniforms & uniforms[[buffer(2)]],
    uint gid[[thread_position_in_grid]],
    uint lid[[thread_position_in_threadgroup]],
    uint tgid[[threadgroup_position_in_grid]]
) {
    uint num_butterflies = uniforms.n / 2;
    if (gid >= num_butterflies) return;

    uint stage = uniforms.stage;
    uint stride = 1u << stage;
    uint m = 1u << (stage + 1);

    // Thread assignment for coalesced access
    uint pair_idx = gid;
    uint block_idx = pair_idx / stride;
    uint k = pair_idx % stride;

    uint j = block_idx * m + k;
    uint i = j + stride;

    // Twiddle factor with stride
    uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
    ulong w = twiddles[twiddle_idx];

    // Load, compute, store
    ulong u = data[j];
    ulong v = gl_mul(data[i], w);

    data[j] = gl_add(u, v);
    data[i] = gl_sub(u, v);
}

// Batch NTT - process multiple small NTTs in parallel
// Useful when doing many small FFTs (like in polynomial multiplication)
kernel void ntt_batch_butterfly(
    device ulong * data[[buffer(0)]],
    constant ulong * twiddles[[buffer(1)]],
    constant NTTUniforms & uniforms[[buffer(2)]],
    constant uint & batch_count[[buffer(3)]],
    uint gid[[thread_position_in_grid]]
) {
    uint batch_butterflies = uniforms.n / 2;
    uint total_butterflies = batch_butterflies * batch_count;

    if (gid >= total_butterflies) return;

    uint batch_idx = gid / batch_butterflies;
    uint local_gid = gid % batch_butterflies;

    uint stage = uniforms.stage;
    uint stride = 1u << stage;
    uint m = 1u << (stage + 1);

    uint k = local_gid % stride;
    uint j = 2 * stride * (local_gid / stride) + k;
    uint i = j + stride;

    // Offset for this batch
    uint batch_offset = batch_idx * uniforms.n;
    j += batch_offset;
    i += batch_offset;

    uint twiddle_idx = k * (uniforms.n / m) * uniforms.twiddle_stride;
    ulong w = twiddles[twiddle_idx];

    ulong u = data[j];
    ulong v = gl_mul(data[i], w);

    data[j] = gl_add(u, v);
    data[i] = gl_sub(u, v);
}
