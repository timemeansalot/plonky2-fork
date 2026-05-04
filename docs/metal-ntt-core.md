# Metal NTT Core Benchmark

This document describes the `metal_ntt_core` Criterion benchmark, which compares the Metal batched NTT core against the Plonky2 CPU FFT/NTT core.

## What It Measures

Benchmark file:

```text
plonky2/benches/metal_ntt_core.rs
```

The benchmark compares two paths over the same deterministic Goldilocks inputs:

| Case | Implementation | Default workload |
|------|----------------|------------------|
| `Metal NTT batch` | Metal batched forward NTT through `NTT_RUNTIME.batch_ntt_in_place` | `200 x 2^16` through `200 x 2^20` |
| `CPU FFT single-thread batch` | Plonky2 `PolynomialCoeffs::fft_with_options(None, None)` in a sequential Rust loop | `200 x 2^16` through `200 x 2^20` |

The CPU baseline is intentionally single-threaded. This matches the Zeknox `gpu_fft_batch.rs` demo style and answers the core question: one Metal GPU batched NTT path versus one CPU thread running Plonky2 FFT repeatedly.

The benchmark validates Metal output against CPU output before timing each size. It accepts either natural-order output or bit-reversed output, matching the existing Metal NTT correctness tests.

## What It Does Not Measure

This is a core NTT benchmark, not a full prover benchmark.

It does not include:

- full Plonky2 proof generation
- Merkle tree construction
- FRI oracle transposition
- random input generation
- Metal shader loading and runtime initialization inside each timing loop
- host-to-GPU transfer as a separate end-to-end pipeline cost

The Metal timing does include `command_buffer.wait_until_completed()`, so the measured time is GPU completion time, not only command enqueue time.

## Code Paths

The Criterion bench uses a doc-hidden helper:

```text
plonky2/src/hash/metal_ntt_bench.rs
```

That helper wraps the crate-private Metal runtime without making the full `hash::metal` module public.

The helper calls benchmark-only methods added to:

```text
plonky2/src/hash/metal/ntt.rs
```

Those methods allocate a shared Metal buffer, run the existing private batched NTT kernel path, read the flat result back for correctness checks, and track buffer deallocation.

## Prerequisites

Run on macOS with Apple Silicon and Xcode command line tools installed.

Use the same nightly used by the current benchmark work:

```bash
cargo +nightly-2025-12-06 --version
rustc +nightly-2025-12-06 --version
```

Run commands from the repository root after cloning:

```bash
cd plonky2-ntt
```

Do not use machine-specific absolute paths in benchmark commands.

## Build Check

Compile the benchmark without running it:

```bash
RUSTFLAGS="-Ctarget-cpu=native" \
cargo +nightly-2025-12-06 bench --features metal -p plonky2 --bench metal_ntt_core --no-run
```

## Full Benchmark

Run the default benchmark:

```bash
RUSTFLAGS="-Ctarget-cpu=native" \
cargo +nightly-2025-12-06 bench --features metal -p plonky2 --bench metal_ntt_core
```

Default settings:

```text
METAL_NTT_BENCH_LOGS=16,17,18,19,20
METAL_NTT_BENCH_BATCHES=200
```

Criterion will print results like:

```text
metal_ntt_core/Metal NTT batch/200 x 2^16
metal_ntt_core/CPU FFT single-thread batch/200 x 2^16
```

Compute speedup as:

```text
CPU time / Metal time
```

For example, from a quick smoke run:

```text
Metal NTT batch/4 x 2^16:              644.71 us
CPU FFT single-thread batch/4 x 2^16:  3.3435 ms
Speedup: 3.3435 ms / 0.64471 ms = 5.19x
```

## Quick Smoke Run

Use environment variables to reduce the workload while checking that both paths compile, run, and pass the correctness check:

```bash
METAL_NTT_BENCH_LOGS=16 \
METAL_NTT_BENCH_BATCHES=4 \
RUSTFLAGS="-Ctarget-cpu=native" \
cargo +nightly-2025-12-06 bench --features metal -p plonky2 --bench metal_ntt_core -- "2\\^16"
```

`METAL_NTT_BENCH_LOGS` accepts comma-separated log sizes:

```bash
METAL_NTT_BENCH_LOGS=16,18,20 \
METAL_NTT_BENCH_BATCHES=64 \
RUSTFLAGS="-Ctarget-cpu=native" \
cargo +nightly-2025-12-06 bench --features metal -p plonky2 --bench metal_ntt_core
```

## Interpreting Results

Use this benchmark to understand raw Metal NTT kernel throughput against a single CPU thread.

Do not use it alone to decide whether Metal NTT improves proving. The full prover currently keeps Metal NTT routing disabled because end-to-end d13-d17 prove benchmarks showed that CPU Rayon FFT plus Metal Merkle hashing was faster than enabling Metal NTT in the FRI oracle path.

For full prover comparisons, use:

```bash
RUSTFLAGS="-Ctarget-cpu=native" \
cargo +nightly-2025-12-06 test --features metal --release \
  --test bench_degrees -- fast --nocapture --test-threads=1
```

See also:

- `docs/metal-ntt-implementation.md`
- `docs/ntt-benchmark-iterations.md`
