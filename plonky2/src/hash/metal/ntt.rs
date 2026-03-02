//! Metal GPU NTT (Number Theoretic Transform) implementation
//!
//! Provides GPU-accelerated NTT and INTT operations for the Goldilocks field.
//! Based on zeknox CUDA patterns, adapted for Apple Metal.

use std::sync::Mutex;

use metal::*;
use once_cell::sync::Lazy;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::types::Field;

use crate::hash::metal::tracking::{track_allocation, track_deallocation};

/// Goldilocks prime: p = 2^64 - 2^32 + 1
const GOLDILOCKS_PRIME: u64 = 18446744069414584321u64;

/// Goldilocks POWER_OF_TWO_GENERATOR - this is the specific generator used by plonky2
/// It generates a subgroup of order 2^32 in the multiplicative group
const GOLDILOCKS_POWER_OF_TWO_GENERATOR: u64 = 1753635133440165772u64;

/// Two-adicity of Goldilocks field (order of multiplicative group has factor 2^32)
const GOLDILOCKS_TWO_ADICITY: usize = 32;

/// Maximum supported NTT size as log2 (2^24 = 16M elements)
const MAX_LOG_N: usize = 24;

/// Goldilocks coset shift = MULTIPLICATIVE_GROUP_GENERATOR = 7
const GOLDILOCKS_COSET_SHIFT: u64 = 7u64;

/// Uniforms for NTT kernels - must match Metal shader
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NTTUniforms {
    n: u32,
    log_n: u32,
    stage: u32,
    direction: u32,
    twiddle_stride: u32, // Stride in the twiddle table for this NTT size
}

/// NTT shader library
const NTT_SHADERLIB: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/ntt_goldilocks.metallib"
));

/// Metal NTT runtime
pub struct MetalNTT {
    device: Mutex<Device>,
    command_queue: CommandQueue,
    f_bit_reverse: Function,
    f_butterfly: Function,
    f_butterfly_inv: Function,
    f_scale: Function,
    f_batch_bit_reverse: Function,
    f_batch_butterfly: Function,
    twiddle_factors: Buffer,
    inv_twiddle_factors: Buffer,
    n_inverses: Buffer,
    max_log_n: usize,
}

unsafe impl Sync for MetalNTT {}

/// Singleton NTT runtime instance
pub static NTT_RUNTIME: Lazy<MetalNTT> = Lazy::new(|| {
    let device = Device::system_default().unwrap();
    let lib = device.new_library_with_data(NTT_SHADERLIB).unwrap();
    let command_queue = device.new_command_queue();

    let f_bit_reverse = lib.get_function("ntt_bit_reverse", None).unwrap();
    let f_butterfly = lib.get_function("ntt_butterfly", None).unwrap();
    let f_butterfly_inv = lib.get_function("intt_butterfly", None).unwrap();
    let f_scale = lib.get_function("ntt_scale", None).unwrap();
    let f_batch_bit_reverse = lib.get_function("ntt_batch_bit_reverse", None).unwrap();
    let f_batch_butterfly = lib.get_function("ntt_batch_butterfly", None).unwrap();

    // Precompute twiddle factors for all supported sizes
    let (twiddles, inv_twiddles, n_invs) = precompute_twiddles(&device, MAX_LOG_N);

    MetalNTT {
        device: Mutex::new(Device::system_default().unwrap()),
        command_queue,
        f_bit_reverse,
        f_butterfly,
        f_butterfly_inv,
        f_scale,
        f_batch_bit_reverse,
        f_batch_butterfly,
        twiddle_factors: twiddles,
        inv_twiddle_factors: inv_twiddles,
        n_inverses: n_invs,
        max_log_n: MAX_LOG_N,
    }
});

/// Compute a^exp mod m using binary exponentiation
fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, modulus);
        }
        exp >>= 1;
        base = mul_mod(base, base, modulus);
    }
    result
}

/// Compute a * b mod m using 128-bit intermediate
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    let product = (a as u128) * (b as u128);
    (product % (m as u128)) as u64
}

/// Compute modular inverse using Fermat's little theorem
/// a^(-1) = a^(p-2) (mod p) for prime p
fn mod_inverse(a: u64, p: u64) -> u64 {
    pow_mod(a, p - 2, p)
}

/// Compute primitive n-th root of unity for Goldilocks field
/// Uses the same formula as plonky2: POWER_OF_TWO_GENERATOR^(2^(32-log_n))
fn compute_primitive_root(n: usize) -> u64 {
    let log_n = n.trailing_zeros() as usize;
    assert!(
        log_n <= GOLDILOCKS_TWO_ADICITY,
        "n exceeds field's two-adicity"
    );

    // Compute POWER_OF_TWO_GENERATOR^(2^(TWO_ADICITY - log_n))
    // This is done by squaring (TWO_ADICITY - log_n) times
    let mut root = GOLDILOCKS_POWER_OF_TWO_GENERATOR;
    for _ in 0..(GOLDILOCKS_TWO_ADICITY - log_n) {
        root = mul_mod(root, root, GOLDILOCKS_PRIME);
    }
    root
}

/// Precompute all twiddle factors for NTT sizes up to 2^max_log_n
fn precompute_twiddles(device: &Device, max_log_n: usize) -> (Buffer, Buffer, Buffer) {
    let max_n = 1usize << max_log_n;

    // We need twiddle factors for each size
    // For size n, we need w^0, w^1, ..., w^(n/2-1)
    // Total: sum(2^(k-1) for k in 1..=max_log_n) = 2^max_log_n - 1

    let total_twiddles = max_n;
    let mut twiddles = Vec::with_capacity(total_twiddles);
    let mut inv_twiddles = Vec::with_capacity(total_twiddles);

    // Compute twiddles for largest size, smaller sizes use subsets
    let omega = compute_primitive_root(max_n);
    let omega_inv = mod_inverse(omega, GOLDILOCKS_PRIME);

    let mut omega_pow = 1u64;
    let mut omega_inv_pow = 1u64;

    for _ in 0..max_n {
        twiddles.push(omega_pow);
        inv_twiddles.push(omega_inv_pow);
        omega_pow = mul_mod(omega_pow, omega, GOLDILOCKS_PRIME);
        omega_inv_pow = mul_mod(omega_inv_pow, omega_inv, GOLDILOCKS_PRIME);
    }

    // Compute n^(-1) for each size
    let mut n_inverses = Vec::with_capacity(max_log_n + 1);
    for log_n in 0..=max_log_n {
        let n = 1u64 << log_n;
        n_inverses.push(mod_inverse(n, GOLDILOCKS_PRIME));
    }

    // Create Metal buffers with allocation tracking
    let twiddles_size = total_twiddles * std::mem::size_of::<u64>();
    track_allocation(twiddles_size);
    let twiddles_buf = device.new_buffer_with_data(
        twiddles.as_ptr() as *const _,
        twiddles_size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let inv_twiddles_size = total_twiddles * std::mem::size_of::<u64>();
    track_allocation(inv_twiddles_size);
    let inv_twiddles_buf = device.new_buffer_with_data(
        inv_twiddles.as_ptr() as *const _,
        inv_twiddles_size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let n_inverses_size = (max_log_n + 1) * std::mem::size_of::<u64>();
    track_allocation(n_inverses_size);
    let n_inverses_buf = device.new_buffer_with_data(
        n_inverses.as_ptr() as *const _,
        n_inverses_size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Note: These are static buffers that live for the program's duration,
    // so we don't track deallocation (they're never freed)
    (twiddles_buf, inv_twiddles_buf, n_inverses_buf)
}

impl MetalNTT {
    /// Get singleton instance
    pub fn get() -> &'static MetalNTT {
        &NTT_RUNTIME
    }

    /// Warm up the NTT runtime (trigger lazy initialization)
    pub fn warm_up(&self) {
        // Just accessing self triggers Lazy initialization
    }

    /// Compute forward NTT (in-place)
    /// Input: coefficients in natural order
    /// Output: evaluations in bit-reversed order (then reordered)
    pub fn ntt(&self, coeffs: &[GoldilocksField]) -> Vec<GoldilocksField> {
        let n = coeffs.len();
        assert!(n.is_power_of_two(), "NTT size must be power of 2");
        let log_n = n.trailing_zeros() as usize;
        assert!(log_n <= self.max_log_n, "NTT size exceeds maximum supported");

        // Copy input to GPU buffer with tracking
        let buffer_size = n * std::mem::size_of::<u64>();
        track_allocation(buffer_size);
        let data_buffer = self.device.lock().unwrap().new_buffer_with_data(
            coeffs.as_ptr() as *const _,
            buffer_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Perform NTT
        self.ntt_in_place(&data_buffer, n, log_n, false);

        // Read results
        let ptr = data_buffer.contents() as *const GoldilocksField;
        let result = unsafe { std::slice::from_raw_parts(ptr, n).to_vec() };

        // Track deallocation when buffer goes out of scope
        track_deallocation(data_buffer.length() as usize);
        result
    }

    /// Compute inverse NTT (in-place)
    /// Input: evaluations
    /// Output: coefficients
    pub fn intt(&self, evals: &[GoldilocksField]) -> Vec<GoldilocksField> {
        let n = evals.len();
        assert!(n.is_power_of_two(), "INTT size must be power of 2");
        let log_n = n.trailing_zeros() as usize;
        assert!(log_n <= self.max_log_n, "INTT size exceeds maximum supported");

        // Copy input to GPU buffer with tracking
        let buffer_size = n * std::mem::size_of::<u64>();
        track_allocation(buffer_size);
        let data_buffer = self.device.lock().unwrap().new_buffer_with_data(
            evals.as_ptr() as *const _,
            buffer_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Perform INTT
        self.ntt_in_place(&data_buffer, n, log_n, true);

        // Read results
        let ptr = data_buffer.contents() as *const GoldilocksField;
        let result = unsafe { std::slice::from_raw_parts(ptr, n).to_vec() };

        // Track deallocation when buffer goes out of scope
        track_deallocation(data_buffer.length() as usize);
        result
    }

    /// Perform NTT/INTT in place on a buffer
    fn ntt_in_place(&self, data_buffer: &Buffer, n: usize, log_n: usize, inverse: bool) {
        let device = self.device.lock().unwrap();

        // Create pipeline states
        let pipeline_bit_reverse = device
            .new_compute_pipeline_state_with_function(&self.f_bit_reverse)
            .unwrap();

        let pipeline_butterfly = if inverse {
            device
                .new_compute_pipeline_state_with_function(&self.f_butterfly_inv)
                .unwrap()
        } else {
            device
                .new_compute_pipeline_state_with_function(&self.f_butterfly)
                .unwrap()
        };

        let pipeline_scale = device
            .new_compute_pipeline_state_with_function(&self.f_scale)
            .unwrap();

        drop(device); // Release lock before command buffer operations

        let command_buffer = self.command_queue.new_command_buffer();

        // Bit-reversal permutation
        {
            let uniforms = NTTUniforms {
                n: n as u32,
                log_n: log_n as u32,
                stage: 0,
                direction: if inverse { 1 } else { 0 },
                twiddle_stride: 0, // Not used in bit-reversal
            };

            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&pipeline_bit_reverse);
            encoder.set_buffer(0, Some(data_buffer), 0);
            encoder.set_bytes(
                1,
                std::mem::size_of::<NTTUniforms>() as u64,
                &uniforms as *const _ as *const _,
            );

            let threads_per_group = pipeline_bit_reverse.thread_execution_width() as usize;
            let num_groups = (n + threads_per_group - 1) / threads_per_group;
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: num_groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_group as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        }

        // Butterfly stages
        // Twiddle stride calculation: for a given n, we use twiddles spaced by max_n/n
        // So twiddles[i * twiddle_stride] gives omega_n^i where omega_n is the n-th root
        // of unity
        let twiddle_stride = (1 << self.max_log_n) / n;

        let twiddle_buffer = if inverse {
            &self.inv_twiddle_factors
        } else {
            &self.twiddle_factors
        };

        for stage in 0..log_n {
            let uniforms = NTTUniforms {
                n: n as u32,
                log_n: log_n as u32,
                stage: stage as u32,
                direction: if inverse { 1 } else { 0 },
                twiddle_stride: twiddle_stride as u32,
            };

            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&pipeline_butterfly);
            encoder.set_buffer(0, Some(data_buffer), 0);
            encoder.set_buffer(1, Some(twiddle_buffer), 0); // No offset - stride handled in shader
            encoder.set_bytes(
                2,
                std::mem::size_of::<NTTUniforms>() as u64,
                &uniforms as *const _ as *const _,
            );

            let num_butterflies = n / 2;
            let threads_per_group = pipeline_butterfly.thread_execution_width() as usize;
            let num_groups = (num_butterflies + threads_per_group - 1) / threads_per_group;
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: num_groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_group as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        }

        // Scale by n^(-1) for inverse NTT
        if inverse {
            let encoder = command_buffer
                .compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
            encoder.set_compute_pipeline_state(&pipeline_scale);
            encoder.set_buffer(0, Some(data_buffer), 0);
            encoder.set_buffer(
                1,
                Some(&self.n_inverses),
                (log_n * std::mem::size_of::<u64>()) as u64,
            );
            encoder.set_bytes(2, 4, &(n as u32) as *const _ as *const _);

            let threads_per_group = pipeline_scale.thread_execution_width() as usize;
            let num_groups = (n + threads_per_group - 1) / threads_per_group;
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: num_groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_group as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Perform Low Degree Extension (LDE) on polynomial values
    ///
    /// Takes evaluations of a polynomial on a subgroup H and returns evaluations
    /// on a larger subgroup H' of size |H| * 2^rate_bits.
    ///
    /// Process: IFFT(values) -> zero-pad -> FFT
    pub fn lde(&self, values: &[GoldilocksField], rate_bits: usize) -> Vec<GoldilocksField> {
        let n = values.len();
        assert!(n.is_power_of_two(), "LDE input size must be power of 2");
        let extended_n = n << rate_bits;
        let extended_log_n = extended_n.trailing_zeros() as usize;
        assert!(
            extended_log_n <= self.max_log_n,
            "Extended size exceeds maximum supported"
        );

        // IFFT to get coefficients
        let coeffs = self.intt(values);

        // Zero-pad coefficients to extended size
        let mut extended_coeffs = coeffs;
        extended_coeffs.resize(extended_n, GoldilocksField::ZERO);

        // FFT on extended coefficients
        self.ntt(&extended_coeffs)
    }

    /// Perform Low Degree Extension onto a coset
    ///
    /// Similar to lde(), but evaluates on the coset shift*H' instead of H'.
    /// The coset shift is F::coset_shift() = F::MULTIPLICATIVE_GROUP_GENERATOR
    pub fn lde_onto_coset(
        &self,
        values: &[GoldilocksField],
        rate_bits: usize,
    ) -> Vec<GoldilocksField> {
        let n = values.len();
        assert!(n.is_power_of_two(), "LDE input size must be power of 2");
        let extended_n = n << rate_bits;
        let extended_log_n = extended_n.trailing_zeros() as usize;
        assert!(
            extended_log_n <= self.max_log_n,
            "Extended size exceeds maximum supported"
        );

        // IFFT to get coefficients
        let coeffs = self.intt(values);

        // Zero-pad coefficients to extended size
        let mut extended_coeffs = coeffs;
        extended_coeffs.resize(extended_n, GoldilocksField::ZERO);

        // Coset FFT - multiply coefficients by shift^i then do FFT
        // For Goldilocks, coset_shift = MULTIPLICATIVE_GROUP_GENERATOR = 7
        self.coset_ntt(&extended_coeffs, GOLDILOCKS_COSET_SHIFT)
    }

    /// Coset NTT: evaluates polynomial on coset shift*H instead of H
    ///
    /// This is done by multiplying coefficients by shift^i before doing NTT:
    /// P(shift * omega^j) = NTT(c[i] * shift^i)
    pub fn coset_ntt(&self, coeffs: &[GoldilocksField], shift: u64) -> Vec<GoldilocksField> {
        let n = coeffs.len();
        assert!(n.is_power_of_two(), "Coset NTT size must be power of 2");

        // Multiply coefficients by shift^i
        let mut shifted_coeffs = Vec::with_capacity(n);
        let mut shift_pow = 1u64;
        for &c in coeffs {
            let shifted = mul_mod(c.0, shift_pow, GOLDILOCKS_PRIME);
            shifted_coeffs.push(GoldilocksField::from_canonical_u64(shifted));
            shift_pow = mul_mod(shift_pow, shift, GOLDILOCKS_PRIME);
        }

        // Do regular NTT
        self.ntt(&shifted_coeffs)
    }

    /// Coset INTT: inverse of coset_ntt
    ///
    /// Takes evaluations on coset shift*H and returns coefficients
    pub fn coset_intt(&self, evals: &[GoldilocksField], shift: u64) -> Vec<GoldilocksField> {
        let n = evals.len();
        assert!(n.is_power_of_two(), "Coset INTT size must be power of 2");

        // Do regular INTT first
        let mut coeffs = self.intt(evals);

        // Multiply coefficients by shift^(-i)
        let shift_inv = mod_inverse(shift, GOLDILOCKS_PRIME);
        let mut shift_inv_pow = 1u64;
        for c in coeffs.iter_mut() {
            let unshifted = mul_mod(c.0, shift_inv_pow, GOLDILOCKS_PRIME);
            *c = GoldilocksField::from_canonical_u64(unshifted);
            shift_inv_pow = mul_mod(shift_inv_pow, shift_inv, GOLDILOCKS_PRIME);
        }

        coeffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plonky2_field::types::Field64;

    #[test]
    fn test_pow_mod() {
        // Test: 7^2 mod 13 = 49 mod 13 = 10
        assert_eq!(pow_mod(7, 2, 13), 10);
        // Test: identity
        assert_eq!(pow_mod(5, 0, 7), 1);
        assert_eq!(pow_mod(5, 1, 7), 5);
    }

    #[test]
    fn test_mod_inverse() {
        // For prime p, a * a^(-1) = 1 (mod p)
        let p = GOLDILOCKS_PRIME;
        let a = 12345678901234567u64;
        let a_inv = mod_inverse(a, p);
        assert_eq!(mul_mod(a, a_inv, p), 1);
    }

    #[test]
    fn test_primitive_root() {
        // Test that w^n = 1 (mod p)
        let n = 256;
        let omega = compute_primitive_root(n);
        assert_eq!(pow_mod(omega, n as u64, GOLDILOCKS_PRIME), 1);

        // Test that w^(n/2) != 1 (mod p) for proper primitive root
        assert_ne!(pow_mod(omega, (n / 2) as u64, GOLDILOCKS_PRIME), 1);

        // Verify it matches plonky2's formula
        // For n=256: omega = g^(2^(32-8)) = g^(2^24)
        let expected = {
            let mut r = GOLDILOCKS_POWER_OF_TWO_GENERATOR;
            for _ in 0..24 {
                r = mul_mod(r, r, GOLDILOCKS_PRIME);
            }
            r
        };
        assert_eq!(omega, expected);
    }

    // ---- NTT / INTT / LDE correctness tests ----

    fn reverse_bits_usize(x: usize, bits: usize) -> usize {
        let mut result = 0;
        let mut val = x;
        for _ in 0..bits {
            result = (result << 1) | (val & 1);
            val >>= 1;
        }
        result
    }

    /// Compare Metal NTT against CPU FFT at a given size.
    /// CPU FFT returns natural order; Metal NTT may return bit-reversed order.
    /// We try both orderings and use whichever matches.
    fn test_ntt_at_size(log_n: usize) {
        use plonky2_field::fft::fft;
        use plonky2_field::polynomial::PolynomialCoeffs;

        let n = 1usize << log_n;
        let coeffs: Vec<GoldilocksField> = (0..n)
            .map(|i| GoldilocksField::from_canonical_u64(i as u64 % GoldilocksField::ORDER))
            .collect();

        // CPU FFT: returns PolynomialValues in natural order
        let poly = PolynomialCoeffs::new(coeffs.clone());
        let cpu_result = fft(poly);

        // Metal NTT
        let gpu_result = NTT_RUNTIME.ntt(&coeffs);

        // First check if GPU output is in natural order (same as CPU)
        let natural_match = (0..n).all(|i| gpu_result[i] == cpu_result.values[i]);

        if natural_match {
            println!(
                "NTT log_n={}: GPU output is in natural order, all {} values match",
                log_n, n
            );
            return;
        }

        // Otherwise check if GPU output is in bit-reversed order
        for i in 0..n {
            let rev_i = reverse_bits_usize(i, log_n);
            assert_eq!(
                gpu_result[i], cpu_result.values[rev_i],
                "NTT mismatch at index {} (bit-reversed {}), log_n={}",
                i, rev_i, log_n
            );
        }
        println!(
            "NTT log_n={}: GPU output is in bit-reversed order, all {} values match after reorder",
            log_n, n
        );
    }

    #[test]
    fn test_ntt_2_16() {
        test_ntt_at_size(16);
    }

    #[test]
    fn test_ntt_2_20() {
        test_ntt_at_size(20);
    }

    /// Test INTT roundtrip: NTT then INTT should recover the original coefficients
    fn test_intt_roundtrip_at_size(log_n: usize) {
        let n = 1usize << log_n;
        let original: Vec<GoldilocksField> = (0..n)
            .map(|i| GoldilocksField::from_canonical_u64(i as u64 % GoldilocksField::ORDER))
            .collect();

        let ntt_result = NTT_RUNTIME.ntt(&original);
        let recovered = NTT_RUNTIME.intt(&ntt_result);

        for i in 0..n {
            assert_eq!(
                recovered[i], original[i],
                "INTT roundtrip mismatch at index {}, log_n={}: got {:?}, expected {:?}",
                i, log_n, recovered[i], original[i]
            );
        }
        println!(
            "INTT roundtrip log_n={}: all {} values match",
            log_n, n
        );
    }

    #[test]
    fn test_intt_roundtrip_2_16() {
        test_intt_roundtrip_at_size(16);
    }

    #[test]
    fn test_intt_roundtrip_2_20() {
        test_intt_roundtrip_at_size(20);
    }

    /// Compare Metal lde_onto_coset against CPU path:
    ///   CPU: ifft(values) -> lde(rate_bits) -> coset_fft(coset_shift)
    ///   GPU: intt(values) -> zero_pad -> coset_ntt(shift)
    fn test_lde_onto_coset_at_size(log_n: usize, rate_bits: usize) {
        use plonky2_field::fft::ifft;
        use plonky2_field::polynomial::PolynomialValues;

        let n = 1usize << log_n;
        let extended_n = n << rate_bits;
        let extended_log_n = extended_n.trailing_zeros() as usize;

        let values: Vec<GoldilocksField> = (0..n)
            .map(|i| GoldilocksField::from_canonical_u64((i as u64 + 1) % GoldilocksField::ORDER))
            .collect();

        // CPU path: ifft -> lde (zero-pad) -> coset_fft
        let poly_values = PolynomialValues::new(values.clone());
        let cpu_coeffs = ifft(poly_values);
        let cpu_lde_coeffs = cpu_coeffs.lde(rate_bits);
        let cpu_result = cpu_lde_coeffs.coset_fft(GoldilocksField::coset_shift());

        // GPU path
        let gpu_result = NTT_RUNTIME.lde_onto_coset(&values, rate_bits);

        // First check natural order match
        let natural_match = (0..extended_n).all(|i| gpu_result[i] == cpu_result.values[i]);

        if natural_match {
            println!(
                "LDE onto coset (log_n={}, rate_bits={}): GPU output is in natural order, all {} values match",
                log_n, rate_bits, extended_n
            );
            return;
        }

        // Check bit-reversed order match
        let bit_reversed_match = (0..extended_n).all(|i| {
            let rev_i = reverse_bits_usize(i, extended_log_n);
            gpu_result[i] == cpu_result.values[rev_i]
        });

        if bit_reversed_match {
            println!(
                "LDE onto coset (log_n={}, rate_bits={}): GPU output is in bit-reversed order, all {} values match after reorder",
                log_n, rate_bits, extended_n
            );
            return;
        }

        // Neither matched - show first few mismatches for debugging
        let mut mismatches = 0;
        for i in 0..extended_n {
            if gpu_result[i] != cpu_result.values[i] {
                if mismatches < 10 {
                    let rev_i = reverse_bits_usize(i, extended_log_n);
                    println!(
                        "  Mismatch at i={}: gpu={:?}, cpu_natural={:?}, cpu_bitrev={:?}",
                        i, gpu_result[i], cpu_result.values[i], cpu_result.values[rev_i]
                    );
                }
                mismatches += 1;
            }
        }
        panic!(
            "LDE onto coset (log_n={}, rate_bits={}): {} / {} values mismatched in both natural and bit-reversed order",
            log_n, rate_bits, mismatches, extended_n
        );
    }

    #[test]
    fn test_lde_onto_coset_2_16_rate_2() {
        test_lde_onto_coset_at_size(16, 2);
    }

    #[test]
    fn test_lde_onto_coset_2_18_rate_3() {
        test_lde_onto_coset_at_size(18, 3);
    }
}
