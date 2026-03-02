# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Plonky2 is a SNARK implementation based on PLONK and FRI. This is an OKX fork with additional features including CUDA/GPU support, batch operations, and specialized elliptic curve operations (ECDSA, ECGFP5). The core field is Goldilocks (2^64 - 2^32 + 1).

**Toolchain:** Rust nightly-2024-01-16 (pinned in `rust-toolchain.toml`)

## Build & Test Commands

```bash
# Build (release)
cargo build --release

# Build with native CPU optimizations (recommended for benchmarks)
RUSTFLAGS=-Ctarget-cpu=native cargo build --release

# Run all workspace tests
cargo test --workspace

# Run a single test by name
cargo test --test fibonacci_test -- test_fibonacci_proof --exact

# Run tests in a specific package
cargo test -p plonky2_field

# Check compilation (what CI does)
cargo check --manifest-path plonky2/Cargo.toml
cargo check --manifest-path starky/Cargo.toml

# Format (CI enforces this)
cargo fmt --all --check
cargo fmt --all          # to fix

# Benchmarks
cargo bench --bench merkle
cargo bench --bench field_arithmetic
cargo bench --bench ffts
cargo bench --bench hashing

# Recursion benchmark example
RUSTFLAGS=-Ctarget-cpu=native cargo run --release --example bench_recursion -- -vv

# CUDA tests (requires GPU + LD_LIBRARY_PATH set)
./run_proof_tests.sh
cargo test --features=cuda,batch --release

# Metal tests (Apple Silicon)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test metal_proof -- --nocapture --test-threads=1
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test metal_coalesced -- --nocapture --test-threads=1
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  -p plonky2 --lib -- test_merkle_trees_metal_poseidon_g64 --nocapture

# Metal benchmarks (fast: d13-d17 ~45s, full: d13-d20 ~20+ min)
RUSTFLAGS="-Ctarget-cpu=native" cargo test --features metal --release \
  --test bench_degrees -- fast --nocapture --test-threads=1
```

**CI environment flags:** `RUSTFLAGS="-Copt-level=3 -Cdebug-assertions -Coverflow-checks=y -Cdebuginfo=0"`

## Workspace Structure

| Crate | Path | Purpose |
|-------|------|---------|
| `plonky2` | `plonky2/` | Core PLONK proving system |
| `plonky2_field` | `field/` | Field arithmetic (Goldilocks, extensions, FFT, polynomials) |
| `starky` | `starky/` | STARK prover/verifier |
| `plonky2_u32` | `u32/` | U32 arithmetic gates and gadgets for circuits |
| `plonky2_ecdsa` | `ecdsa/` | ECDSA signature verification gadgets (secp256k1) |
| `plonky2_ecgfp5` | `ecgfp5/` | EC-GFP5 elliptic curve operations |
| `plonky2_util` | `util/` | Shared utilities |
| `plonky2_maybe_rayon` | `maybe_rayon/` | Conditional parallelism (rayon behind `parallel` feature) |
| `gen` | `gen/` | Code generation utilities |

Note: `evm/` and `circom/` exist in the repo but are not workspace members.

## Feature Flags (plonky2 crate)

- **`parallel`** (default) — Rayon-based parallelism
- **`std`** (default) — Standard library support
- **`gate_testing`** (default) — Gate testing utilities
- **`cuda`** — CUDA GPU acceleration for Merkle/NTT
- **`metal`** — Metal GPU acceleration for Merkle/NTT (Apple Silicon)
- **`batch`** — Batch processing mode
- **`timing`** — Performance timing instrumentation
- **`no_cuda`** — Explicitly disable CUDA

## Architecture

### Core Type Flow

```
CircuitBuilder<F, D>  →  build()  →  CircuitData<F, C, D>
                                        ├── prove(PartialWitness)  →  ProofWithPublicInputs<F, C, D>
                                        └── verify(ProofWithPublicInputs)  →  Result<()>
```

The generic parameters: `F` = base field (typically `GoldilocksField`), `C` = config implementing `GenericConfig<D>`, `D` = extension degree (typically 2).

### Key Traits

- **`Field`** / **`RichField`** (`field/src/types.rs`) — Field element arithmetic. `GoldilocksField` is the primary implementation.
- **`Extendable<D>`** / **`FieldExtension<D>`** (`field/src/extension/`) — Extension field support (quadratic, quartic, quintic).
- **`GenericConfig<D>`** (`plonky2/src/plonk/config.rs`) — Bundles field + hashers. Standard configs: `PoseidonGoldilocksConfig`, `KeccakGoldilocksConfig`.
- **`Gate<F, D>`** (`plonky2/src/gates/gate.rs`) — Custom constraint gates. Must implement `eval_unfiltered` (native) and `eval_unfiltered_circuit` (recursive).
- **`WitnessGenerator<F, D>`** / **`SimpleGenerator<F, D>`** (`plonky2/src/iop/generator.rs`) — Witness generation from partial inputs.
- **`Hasher<F>`** / **`AlgebraicHasher<F>`** (`plonky2/src/hash/hashing.rs`) — Hash functions. Algebraic hashers (Poseidon) support in-circuit evaluation.
- **`Stark<F, D>`** (`starky/src/stark.rs`) — STARK constraint interface with `eval_packed_generic` and `eval_ext_circuit`.

### Circuit Construction (plonky2/src/)

- **`plonk/circuit_builder.rs`** — Main API. Users add targets, connect constraints, register gates, then call `build()`.
- **`iop/target.rs`** — `Target` = either `Wire(row, col)` or `VirtualTarget{index}`. `BoolTarget` and `ExtensionTarget<D>` are wrappers.
- **`iop/witness.rs`** — `PartialWitness` holds user-provided values; `PartitionWitness` is the full generated witness.
- **`iop/generator.rs`** — `generate_partial_witness()` iteratively runs generators until all targets are filled.
- **`iop/challenger.rs`** — Fiat-Shamir challenger using a duplex sponge.

### Gates (plonky2/src/gates/)

Constraint primitives. Key implementations: `ArithmeticGate` (base field ops), `ArithmeticExtensionGate`, `PoseidonGate` (135 wires), `LookupGate`/`LookupTableGate`, `RandomAccessGate`, `BaseSumGate`, `ExponentiationGate`.

### Gadgets (plonky2/src/gadgets/)

Higher-level building blocks on top of gates: arithmetic, hashing, interpolation, lookup, polynomial ops, range checks, random access, conditional select.

### FRI Protocol (plonky2/src/fri/)

FRI commitment and query protocol. `FriConfig` controls rate bits, cap height, proof-of-work bits, reduction strategy. `oracle.rs` manages polynomial batch commitments.

### Hash (plonky2/src/hash/)

Poseidon (primary algebraic hash), Poseidon2, Keccak, and BN128 Poseidon variant. Merkle trees with configurable cap height in `merkle_tree.rs`.

### Metal GPU Acceleration (plonky2/src/hash/metal/)

Apple Silicon GPU acceleration via the Metal API. Priority: CUDA > Metal > CPU. See [`docs/metal-acceleration-status.md`](docs/metal-acceleration-status.md) for full details.

**Routing:**
- Merkle: GPU for tree_height 13..=20 (Poseidon only), CPU otherwise
- NTT: GPU for log_n + rate_bits >= 16, CPU otherwise

**Key modules:** `runtime.rs` (device/pipelines), `gpu_thread.rs` (dedicated dispatch thread), `merkle.rs` (Merkle hashing), `ntt.rs` (NTT/LDE), `buffer_pool.rs` (buffer reuse).

**Shaders:** Precompiled `.metallib` files in `plonky2/shaders/`, embedded via `include_bytes!()`. Recompile with `xcrun -sdk macosx metal` + `xcrun -sdk macosx metallib` after editing `.metal` sources.

### Recursion (plonky2/src/recursion/)

Verify proofs inside circuits via `recursive_verifier.rs`. Supports conditional verification and cyclic (self-referential) recursion.

### Build Script

`plonky2/build.rs` compiles a Poseidon BN128 C library via Go (clones `go-iden3-crypto` on macOS). Requires Go installed. Links `libposeidon-permute-c` statically.

## Code Style

- **Formatting:** `rustfmt.toml` uses `group_imports = "StdExternalCrate"`, `imports_granularity = "Module"`, `unstable_features = true`
- Run `cargo fmt --all` before committing
- CI runs `cargo fmt --all --check`
