use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::plonk::config::Hasher;

const DEFAULT_LOG_SIZES: &[usize] = &[13, 14, 15, 16];

fn bench_leaf_size() -> usize {
    std::env::var("POSEIDON_BENCH_LEAF_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8)
}

fn bench_sample_size() -> usize {
    std::env::var("POSEIDON_BENCH_SAMPLE_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(20)
}

fn bench_leaf_only() -> bool {
    std::env::var("POSEIDON_BENCH_LEAF_ONLY").is_ok()
}

fn hash_or_noop_cpu(leaf: &[GoldilocksField]) -> HashOut<GoldilocksField> {
    if leaf.len() <= 4 {
        HashOut::from_partial(leaf)
    } else {
        PoseidonHash::hash_no_pad(leaf)
    }
}

fn bench_poseidon_cpu(c: &mut Criterion) {
    if std::env::var("POSEIDON_BENCH_SKIP_CPU").is_ok() {
        return;
    }
    let mut group = c.benchmark_group("poseidon-leaf-cpu");
    group.sample_size(bench_sample_size());

    for &log_size in DEFAULT_LOG_SIZES {
        let leaf_count = 1usize << log_size;
        let leaf_size = bench_leaf_size();
        let input_len = leaf_count * leaf_size;

        let input: Vec<GoldilocksField> = (0..input_len)
            .map(|i| GoldilocksField::from_canonical_u64(i as u64))
            .collect();

        let leaves_2d: Vec<Vec<GoldilocksField>> = input
            .chunks(leaf_size)
            .map(|chunk: &[GoldilocksField]| chunk.to_vec())
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}-leaves", leaf_count)),
            &leaves_2d,
            |b, leaves: &Vec<Vec<GoldilocksField>>| {
                b.iter_batched(
                    || leaves.clone(),
                    |leaves_clone: Vec<Vec<GoldilocksField>>| {
                        let mut out = Vec::with_capacity(leaves_clone.len());
                        for leaf in &leaves_clone {
                            out.push(hash_or_noop_cpu(leaf));
                        }
                        out
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

#[cfg(all(target_os = "macos", any(feature = "metal-linear-merkle", feature = "metal-optimized")))]
fn bench_poseidon_metal(c: &mut Criterion) {
    use plonky2::gpu::metal::poseidon_interleaved::MetalRuntime;

    let mut group = c.benchmark_group("poseidon-leaf-metal");
    group.sample_size(bench_sample_size());

    for &log_size in DEFAULT_LOG_SIZES {
        let leaf_count = 1usize << log_size;
        let leaf_size = bench_leaf_size();
        let input_len = leaf_count * leaf_size;

        let input: Vec<GoldilocksField> = (0..input_len)
            .map(|i| GoldilocksField::from_canonical_u64(i as u64))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}-leaves", leaf_count)),
            &input,
            |b, leaves: &Vec<GoldilocksField>| {
                b.iter_batched(
                    || leaves.clone(),
                    |leaves_clone: Vec<GoldilocksField>| {
                        MetalRuntime::get().poseidon_hash_leaves_linear_only(&leaves_clone, leaf_size)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_poseidon_tree_level_cpu(c: &mut Criterion) {
    if std::env::var("POSEIDON_BENCH_SKIP_CPU").is_ok() {
        return;
    }
    let mut group = c.benchmark_group("poseidon-tree-level-cpu");
    group.sample_size(bench_sample_size());

    for &log_size in DEFAULT_LOG_SIZES {
        let node_count = 1usize << (log_size - 1);
        let input_len = node_count * 8;

        let input: Vec<GoldilocksField> = (0..input_len)
            .map(|i| GoldilocksField::from_canonical_u64(i as u64))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}-nodes", node_count)),
            &input,
            |b, children: &Vec<GoldilocksField>| {
                b.iter_batched(
                    || children.clone(),
                    |children_clone: Vec<GoldilocksField>| {
                        let mut out = Vec::with_capacity(node_count);
                        for i in 0..node_count {
                            let base = i * 8;
                            let left = HashOut::from_partial(&children_clone[base..base + 4]);
                            let right = HashOut::from_partial(&children_clone[base + 4..base + 8]);
                            out.push(PoseidonHash::two_to_one(left, right));
                        }
                        out
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

#[cfg(all(target_os = "macos", any(feature = "metal-linear-merkle", feature = "metal-optimized")))]
fn bench_poseidon_tree_level_metal(c: &mut Criterion) {
    use plonky2::gpu::metal::poseidon_interleaved::MetalRuntime;

    let mut group = c.benchmark_group("poseidon-tree-level-metal");
    group.sample_size(bench_sample_size());

    for &log_size in DEFAULT_LOG_SIZES {
        let node_count = 1usize << (log_size - 1);
        let input_len = node_count * 8;

        let input: Vec<u64> = (0..input_len as u64).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}-nodes", node_count)),
            &input,
            |b, children: &Vec<u64>| {
                b.iter_batched(
                    || children.clone(),
                    |children_clone: Vec<u64>| {
                        MetalRuntime::get().poseidon_hash_tree_level_linear_only(
                            &children_clone,
                            node_count,
                        )
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_poseidon_cpu(c);
    #[cfg(all(target_os = "macos", any(feature = "metal-linear-merkle", feature = "metal-optimized")))]
    bench_poseidon_metal(c);
    if !bench_leaf_only() {
        bench_poseidon_tree_level_cpu(c);
        #[cfg(all(target_os = "macos", any(feature = "metal-linear-merkle", feature = "metal-optimized")))]
        bench_poseidon_tree_level_metal(c);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
