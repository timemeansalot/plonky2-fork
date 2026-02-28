//! Metal GPU acceleration for Poseidon Merkle tree construction.
//!
//! Provides `fill_digests_buf_metal`, which is called from `merkle_tree.rs` when
//! the `metal` feature is enabled and CUDA is not.
//!
//! Only accelerates Poseidon/Goldilocks trees with 2^13–2^20 leaves.
//! All other cases fall through to the CPU implementation.

pub(crate) mod buffer_pool;
pub(crate) mod merkle;
pub(crate) mod ntt;
pub(crate) mod runtime;
pub(crate) mod threadgroup_config;
pub(crate) mod tracking;
pub(crate) mod utils;

pub(crate) use runtime::RUNTIME;
