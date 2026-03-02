//! Targeted Merkle tree build benchmark to measure coalesced shader impact.
//! Only benchmarks the Merkle tree construction (not full prove).
//!
//! Run with: cargo test --features metal --release --test bench_coalesced -- --nocapture

use std::time::Instant;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::hash::merkle_tree::MerkleTree;
use plonky2::hash::poseidon::PoseidonHash;

type F = GoldilocksField;

/// Benchmark Merkle tree construction with deterministic leaves.
/// leaf_size simulates real proving: wire commitments have ~135 columns.
fn bench_merkle(tree_height: usize, leaf_size: usize, cap_height: usize, iterations: usize) {
    let num_leaves = 1usize << tree_height;

    eprintln!(
        "\n=== Merkle tree: height={tree_height}, leaves=2^{tree_height}={num_leaves}, \
         leaf_size={leaf_size}, cap_height={cap_height} ==="
    );
    eprintln!(
        "    total data: {} MB",
        (num_leaves * leaf_size * 8) / (1024 * 1024)
    );

    // Generate deterministic 2D test data
    let leaves_2d: Vec<Vec<F>> = (0..num_leaves)
        .map(|i| {
            (0..leaf_size)
                .map(|j| GoldilocksField((i * leaf_size + j) as u64 * 7 + 13))
                .collect()
        })
        .collect();

    let mut times = Vec::new();
    for i in 0..iterations {
        let leaves_clone = leaves_2d.clone();
        let start = Instant::now();
        let _tree = MerkleTree::<GoldilocksField, PoseidonHash>::new_from_2d(
            leaves_clone,
            cap_height,
        );
        let elapsed = start.elapsed();
        times.push(elapsed);
        eprintln!("  iteration {}: {elapsed:.3?}", i + 1);
    }

    let total: std::time::Duration = times.iter().sum();
    let mean = total / iterations as u32;
    let min = times.iter().min().unwrap();
    let max = times.iter().max().unwrap();
    eprintln!("  RESULT: mean={mean:.3?} min={min:.3?} max={max:.3?} ({iterations} iters)");
}

/// Height 20 (1M leaves) — uses linear+threadgroup on both branches (control)
#[test]
fn bench_merkle_height_20() {
    bench_merkle(20, 135, 4, 3);
}

/// Height 21 (2M leaves) — coalesced on feature, CPU on main
#[test]
fn bench_merkle_height_21() {
    bench_merkle(21, 135, 4, 3);
}

/// Height 22 (4M leaves) — coalesced on feature, CPU on main
#[test]
fn bench_merkle_height_22() {
    bench_merkle(22, 135, 4, 2);
}
