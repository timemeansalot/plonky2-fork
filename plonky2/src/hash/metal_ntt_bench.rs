//! Doc-hidden helpers for Criterion benchmarks of the Metal NTT core.

use core::fmt;

use metal::Buffer;
use plonky2_field::goldilocks_field::GoldilocksField;

use crate::hash::metal::ntt::NTT_RUNTIME;

#[doc(hidden)]
pub fn warm_up() {
    NTT_RUNTIME.warm_up();
}

#[doc(hidden)]
pub struct MetalNttBenchBuffer {
    buffer: Buffer,
    log_n: usize,
    batches: usize,
}

impl fmt::Debug for MetalNttBenchBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalNttBenchBuffer")
            .field("log_n", &self.log_n)
            .field("batches", &self.batches)
            .field("buffer_len", &self.buffer.length())
            .finish()
    }
}

impl MetalNttBenchBuffer {
    #[doc(hidden)]
    pub fn new(log_n: usize, batches: usize, input: &[GoldilocksField]) -> Self {
        let n = 1usize << log_n;
        assert_eq!(
            input.len(),
            n * batches,
            "input must contain exactly batches * 2^log_n elements"
        );

        let buffer = NTT_RUNTIME.bench_buffer_from_coeffs(input);
        Self {
            buffer,
            log_n,
            batches,
        }
    }

    #[doc(hidden)]
    pub fn run_forward_ntt(&mut self) {
        NTT_RUNTIME.bench_forward_batch_ntt(&self.buffer, self.log_n, self.batches);
    }

    #[doc(hidden)]
    pub fn read_flat(&self) -> Vec<GoldilocksField> {
        NTT_RUNTIME.bench_read_buffer(&self.buffer, self.log_n, self.batches)
    }
}

impl Drop for MetalNttBenchBuffer {
    fn drop(&mut self) {
        NTT_RUNTIME.bench_track_buffer_deallocation(&self.buffer);
    }
}
