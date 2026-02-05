/// E2E proving benchmark to compare CPU vs GPU performance
/// Run with:
///   CPU only:  cargo run --release --example bench_e2e_prove -- [ROUNDS] [START] [END]
///   With CUDA: NUM_OF_GPUS=1 cargo run --release --features=cuda --example bench_e2e_prove -- [ROUNDS] [START] [END]
///
/// Arguments:
///   ROUNDS - Number of benchmark rounds (default: 3)
///   START  - Starting log_size (default: 13)
///   END    - Ending log_size (default: 18)
///
/// Examples:
///   cargo run --release --example bench_e2e_prove -- 3 13 18
///   NUM_OF_GPUS=1 cargo run --release --features=cuda --example bench_e2e_prove -- 5 14 20

use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use plonky2::field::goldilocks_field::GoldilocksField;
#[cfg(feature = "cuda")]
use plonky2::field::types::PrimeField64;

#[cfg(not(target_os = "macos"))]
use jemallocator::Jemalloc;

#[cfg(not(target_os = "macos"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn parse_args() -> (usize, usize, usize) {
    let args: Vec<String> = std::env::args().collect();

    let rounds = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let start = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(13);

    let end = args.get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(18);

    (rounds, start, end)
}

#[cfg(feature = "cuda")]
fn init_gpu() {
    use zeknox::{get_number_of_gpus_rs, init_coset_rs, init_twiddle_factors_rs};

    // Check if NUM_OF_GPUS is already set in environment
    let num_of_gpus = match std::env::var("NUM_OF_GPUS") {
        Ok(val) => val.parse().unwrap_or_else(|_| get_number_of_gpus_rs()),
        Err(_) => {
            let detected = get_number_of_gpus_rs();
            std::env::set_var("NUM_OF_GPUS", detected.to_string());
            detected
        }
    };

    println!("Initializing {} GPU(s) for CUDA acceleration...", num_of_gpus);

    // Initialize twiddle factors for log_n from 2 to 26 (covers most circuit sizes)
    let log_ns: Vec<usize> = (2..27).collect();

    for device_id in 0..num_of_gpus {
        // Initialize coset for this GPU
        init_coset_rs(
            device_id,
            26, // max log_n
            GoldilocksField::coset_shift().to_canonical_u64(),
        );

        // Initialize twiddle factors for each log_n
        for log_n in &log_ns {
            init_twiddle_factors_rs(device_id, *log_n);
        }
    }
    println!("GPU initialization complete!");
}

fn main() -> Result<()> {
    // Parse command line arguments
    let (rounds, start, end) = parse_args();

    // Initialize logging
    let _ = env_logger::builder().format_timestamp(None).try_init();

    // Initialize GPU if CUDA feature is enabled
    #[cfg(feature = "cuda")]
    init_gpu();

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    println!("Using PoseidonGoldilocksConfig (Goldilocks 64-bit)");
    println!("Running {} rounds per circuit size", rounds);
    println!("Log size range: {} to {}\n", start, end);

    // Test circuit sizes
    for log_size in start..=end {
        println!("============================================================");
        println!("Circuit size: 2^{} = {} gates", log_size, 1 << log_size);
        println!("============================================================");

        let mut build_times: Vec<Duration> = Vec::new();
        let mut prove_times: Vec<Duration> = Vec::new();
        let mut verify_times: Vec<Duration> = Vec::new();

        for round in 0..rounds {
            println!("\n--- Round {} ---", round + 1);

            let config = CircuitConfig::standard_recursion_config();
            let mut builder = CircuitBuilder::<F, D>::new(config);

            // Create a circuit with many arithmetic operations
            let num_ops = 1 << log_size;

            let initial = builder.add_virtual_target();
            let mut cur = initial;

            // Create a long chain of arithmetic operations
            for i in 0..num_ops {
                let constant = builder.constant(F::from_canonical_u64((i % 1000) as u64 + 1));
                if i % 2 == 0 {
                    cur = builder.mul(cur, constant);
                } else {
                    cur = builder.add(cur, constant);
                }
            }

            // Register public inputs
            builder.register_public_input(initial);
            builder.register_public_input(cur);

            // BUILD (setup)
            let build_start = Instant::now();
            let data = builder.build::<C>();
            let build_time = build_start.elapsed();
            build_times.push(build_time);
            println!("Build: {:?}", build_time);

            // PROVE
            let mut pw = PartialWitness::new();
            pw.set_target(initial, F::from_canonical_u64(12345));

            let prove_start = Instant::now();
            let proof = data.prove(pw)?;
            let prove_time = prove_start.elapsed();
            prove_times.push(prove_time);
            println!("Prove: {:?}", prove_time);

            // VERIFY
            let verify_start = Instant::now();
            data.verify(proof.clone())?;
            let verify_time = verify_start.elapsed();
            verify_times.push(verify_time);
            println!("Verify: {:?}", verify_time);
        }

        // Calculate averages
        let avg_build: Duration = build_times.iter().sum::<Duration>() / rounds as u32;
        let avg_prove: Duration = prove_times.iter().sum::<Duration>() / rounds as u32;
        let avg_verify: Duration = verify_times.iter().sum::<Duration>() / rounds as u32;

        println!("\n>>> AVERAGE for 2^{} ({} gates) <<<", log_size, 1 << log_size);
        println!("  Build:  {:?}", avg_build);
        println!("  Prove:  {:?}", avg_prove);
        println!("  Verify: {:?}", avg_verify);
        println!("  Total:  {:?}", avg_build + avg_prove + avg_verify);
        println!("");
    }

    Ok(())
}
