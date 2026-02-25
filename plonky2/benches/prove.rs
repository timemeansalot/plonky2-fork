//! Prove-time benchmark: CPU vs Metal GPU.
//!
//! ## Usage
//!
//! ```bash
//! # Step 1: record CPU baseline
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench prove -- --save-baseline cpu
//!
//! # Step 2: record Metal numbers and compare against baseline
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --features metal --bench prove -- --baseline cpu
//! ```
//!
//! Criterion writes an HTML report to `target/criterion/prove/` showing
//! the per-degree speedup / regression with confidence intervals.

mod allocator;

use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use plonky2::gates::noop::NoopGate;
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

/// Build a circuit that requires exactly 2^degree_bits rows.
/// Uses (2^(degree_bits-1) + 1) NoopGates so that next_power_of_2 = 2^degree_bits.
fn build_circuit(degree_bits: usize) -> CircuitData<F, C, D> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    for _ in 0..((1usize << (degree_bits - 1)) + 1) {
        builder.add_gate(NoopGate, vec![]);
    }
    builder.build::<C>()
}

fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("prove/poseidon_goldilocks");

    // degree 13: FRI oracle tree_height = 16  → Metal active (range 13-20)
    // degree 17: FRI oracle tree_height = 20  → Metal active (upper edge of range)
    //
    // degree 19 is omitted: ~130s per iteration makes Criterion sampling impractical.
    // Use test_proof_degree19_timing for a single-shot comparison instead.
    //
    // Criterion requires sample_size >= 10.  degree-17 prove() ~6s per call;
    // measurement_time(30s) → ~5 iterations per sample, 10 samples total (~5 min).
    for (degree_bits, samples, budget_secs) in [(13usize, 20usize, 30u64), (17, 10, 30)] {
        let label = format!("degree_{degree_bits}");
        let data = build_circuit(degree_bits);

        group.sample_size(samples);
        group.measurement_time(Duration::from_secs(budget_secs));

        group.bench_with_input(BenchmarkId::new("prove", &label), &label, |b, _| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let pw = PartialWitness::new();
                    let start = Instant::now();
                    let _proof = data.prove(pw).unwrap();
                    total += start.elapsed();
                }
                total
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_prove);
criterion_main!(benches);
