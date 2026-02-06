use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::hash::merkle_tree::MerkleTree;
use plonky2::hash::poseidon::PoseidonHash;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const LEAF_SIZES: &[usize] = &[135];
const LEAF_COUNTS: &[usize] = &[8192, 16384, 32768, 65536, 131072];
const CAP_HEIGHT: usize = 2;
const RUNS: usize = 10;

fn build_leaves_1d(leaf_count: usize, leaf_size: usize) -> Vec<GoldilocksField> {
    let mut leaves = Vec::with_capacity(leaf_count * leaf_size);
    for i in 0..leaf_count {
        for j in 0..leaf_size {
            leaves.push(GoldilocksField::from_canonical_u64((i * leaf_size + j) as u64));
        }
    }
    leaves
}

fn main() {
    let out_dir = Path::new("bench_baselines");
    fs::create_dir_all(out_dir).expect("create bench_baselines");
    let out_path = out_dir.join("merkle_cpu_mean.txt");

    let mut file = fs::File::create(&out_path).expect("create output file");
    writeln!(file, "# leaves cpu_ms (mean of {} runs)", RUNS).ok();

    println!("=== Merkle CPU Baseline ===");
    println!("Output: {}", out_path.display());
    println!("leaf_size(s): {:?}, cap_height: {}", LEAF_SIZES, CAP_HEIGHT);
    println!("leaf_counts: {:?}, runs: {}", LEAF_COUNTS, RUNS);
    println!(
        "CPU SIMD: avx2={}, avx512dq={}",
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx512dq")
    );
    println!();

    for &leaf_size in LEAF_SIZES {
        for &leaf_count in LEAF_COUNTS {
            let leaves = build_leaves_1d(leaf_count, leaf_size);
            let mut total_ms = 0.0;

            for _ in 0..RUNS {
                let start = Instant::now();
                let _ = MerkleTree::<GoldilocksField, PoseidonHash>::new_from_1d(
                    leaves.clone(),
                    leaf_size,
                    CAP_HEIGHT,
                );
                total_ms += start.elapsed().as_secs_f64() * 1000.0;
            }

            let mean_ms = total_ms / RUNS as f64;
            println!(
                "leaves={} leaf_size={} mean_ms={:.3}",
                leaf_count, leaf_size, mean_ms
            );
            writeln!(file, "{} {:.6}", leaf_count, mean_ms).ok();
        }
    }
}
