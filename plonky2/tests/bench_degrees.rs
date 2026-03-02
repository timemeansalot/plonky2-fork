//! Prove timing across degrees.
//!
//! Fast test (d13-d17, ~2 min):
//!   cargo test --features metal --release --test bench_degrees -- fast --nocapture --test-threads=1
//!
//! Full test (d13-d20, ~20+ min):
//!   cargo test --features metal --release --test bench_degrees -- full --nocapture --test-threads=1

use std::time::Instant;

use plonky2::gates::noop::NoopGate;
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

fn bench_degree(degree_bits: usize) {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    for _ in 0..((1usize << (degree_bits - 1)) + 1) {
        builder.add_gate(NoopGate, vec![]);
    }

    let t_build = Instant::now();
    let data = builder.build::<C>();
    let build_time = t_build.elapsed();

    // Warm-up run (not counted)
    let pw = PartialWitness::new();
    let _ = data.prove(pw).unwrap();

    // Timed runs
    let mut times = Vec::new();
    let iters = if degree_bits <= 17 { 3 } else { 2 };
    for _ in 0..iters {
        let pw = PartialWitness::new();
        let start = Instant::now();
        let proof = data.prove(pw).unwrap();
        times.push(start.elapsed());
        // Verify once
        if times.len() == 1 {
            data.verify(proof).unwrap();
        }
    }

    let total: std::time::Duration = times.iter().sum();
    let mean = total / iters as u32;
    let min = times.iter().min().unwrap();
    eprintln!("degree {degree_bits}: build={build_time:.3?} prove_mean={mean:.3?} prove_min={min:.3?} ({iters} iters)");
}

// Fast benchmarks (d13-d17): ~2 min total
#[test] fn fast_d13() { bench_degree(13); }
#[test] fn fast_d14() { bench_degree(14); }
#[test] fn fast_d15() { bench_degree(15); }
#[test] fn fast_d16() { bench_degree(16); }
#[test] fn fast_d17() { bench_degree(17); }

// Full benchmarks include fast + large degrees (d18-d20)
#[test] fn full_d13() { bench_degree(13); }
#[test] fn full_d14() { bench_degree(14); }
#[test] fn full_d15() { bench_degree(15); }
#[test] fn full_d16() { bench_degree(16); }
#[test] fn full_d17() { bench_degree(17); }
#[test] fn full_d18() { bench_degree(18); }
#[test] fn full_d19() { bench_degree(19); }
#[test] fn full_d20() { bench_degree(20); }
