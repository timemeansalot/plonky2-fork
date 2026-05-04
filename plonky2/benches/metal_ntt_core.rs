mod allocator;

#[cfg(feature = "metal")]
use std::env;

#[cfg(feature = "metal")]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "metal")]
use plonky2::field::goldilocks_field::GoldilocksField;
#[cfg(feature = "metal")]
use plonky2::field::polynomial::PolynomialCoeffs;
#[cfg(feature = "metal")]
use plonky2::field::types::{Field, Field64};
#[cfg(feature = "metal")]
use plonky2::hash::metal_ntt_bench::{warm_up, MetalNttBenchBuffer};

#[cfg(feature = "metal")]
const DEFAULT_BATCHES: usize = 200;
#[cfg(feature = "metal")]
const LOG_N_SIZES: [usize; 5] = [16, 17, 18, 19, 20];

#[cfg(feature = "metal")]
fn configured_batches() -> usize {
    env::var("METAL_NTT_BENCH_BATCHES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_BATCHES)
}

#[cfg(feature = "metal")]
fn configured_log_n_sizes() -> Vec<usize> {
    env::var("METAL_NTT_BENCH_LOGS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|part| {
                    part.trim()
                        .parse()
                        .expect("METAL_NTT_BENCH_LOGS must contain comma-separated usize values")
                })
                .collect()
        })
        .unwrap_or_else(|| LOG_N_SIZES.to_vec())
}

#[cfg(feature = "metal")]
fn input_for(log_n: usize, batches: usize) -> Vec<GoldilocksField> {
    let n = 1usize << log_n;
    (0..batches * n)
        .map(|i| {
            let x = ((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)) % GoldilocksField::ORDER;
            GoldilocksField::from_canonical_u64(x)
        })
        .collect()
}

#[cfg(feature = "metal")]
fn cpu_ntt_batch(log_n: usize, batches: usize, input: &[GoldilocksField]) -> Vec<GoldilocksField> {
    let n = 1usize << log_n;
    let mut output = Vec::with_capacity(input.len());
    for batch in 0..batches {
        let start = batch * n;
        let coeffs = PolynomialCoeffs::new(input[start..start + n].to_vec());
        output.extend(coeffs.fft_with_options(None, None).values);
    }
    output
}

#[cfg(feature = "metal")]
fn reverse_bits_usize(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

#[cfg(feature = "metal")]
fn assert_same_ntt_output(
    log_n: usize,
    batches: usize,
    cpu: &[GoldilocksField],
    metal: &[GoldilocksField],
) {
    let n = 1usize << log_n;
    assert_eq!(cpu.len(), metal.len());

    for batch in 0..batches {
        let start = batch * n;
        let cpu_batch = &cpu[start..start + n];
        let metal_batch = &metal[start..start + n];

        if metal_batch == cpu_batch {
            continue;
        }

        for i in 0..n {
            let rev_i = reverse_bits_usize(i, log_n);
            assert_eq!(
                metal_batch[i], cpu_batch[rev_i],
                "Metal NTT mismatch at batch {batch}, index {i}, bit-reversed index {rev_i}, log_n={log_n}"
            );
        }
    }
}

#[cfg(feature = "metal")]
fn verify_metal_ntt(log_n: usize, batches: usize, input: &[GoldilocksField]) {
    let cpu = cpu_ntt_batch(log_n, batches, input);
    let mut metal_buffer = MetalNttBenchBuffer::new(log_n, batches, input);
    metal_buffer.run_forward_ntt();
    let metal = metal_buffer.read_flat();
    assert_same_ntt_output(log_n, batches, &cpu, &metal);
}

#[cfg(feature = "metal")]
fn criterion_benchmark(c: &mut Criterion) {
    warm_up();

    let batches = configured_batches();
    let mut group = c.benchmark_group("metal_ntt_core");
    for log_n in configured_log_n_sizes() {
        let input = input_for(log_n, batches);
        let verification_batches = batches.min(2);
        let verification_len = verification_batches * (1usize << log_n);

        verify_metal_ntt(log_n, verification_batches, &input[..verification_len]);

        group.sample_size(20).bench_with_input(
            BenchmarkId::new("Metal NTT batch", format!("{batches} x 2^{log_n}")),
            &log_n,
            |b, &log_n| {
                let mut metal_buffer = MetalNttBenchBuffer::new(log_n, batches, &input);
                b.iter(|| metal_buffer.run_forward_ntt());
            },
        );

        group.sample_size(10).bench_with_input(
            BenchmarkId::new(
                "CPU FFT single-thread batch",
                format!("{batches} x 2^{log_n}"),
            ),
            &log_n,
            |b, &log_n| {
                b.iter(|| cpu_ntt_batch(log_n, batches, &input));
            },
        );
    }
    group.finish();
}

#[cfg(feature = "metal")]
criterion_group!(benches, criterion_benchmark);
#[cfg(feature = "metal")]
criterion_main!(benches);

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("metal_ntt_core requires --features metal");
}
