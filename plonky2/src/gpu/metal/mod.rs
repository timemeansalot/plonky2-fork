// Core modules
pub mod tracking;
pub mod buffer_pool;
pub mod merkle_job;
pub mod threadgroup_config;
pub mod utils;
pub mod runtime;

// Merkle tree implementation modules
pub mod merkle_recursive;

#[cfg(feature = "metal-threadgroup")]
pub mod merkle_threadgroup;

#[cfg(any(feature = "metal-linear-merkle", feature = "metal-optimized"))]
pub mod merkle_linear;

#[cfg(feature = "metal-optimized")]
pub mod merkle_linear_threadgroup;

#[cfg(feature = "metal-bandwidth-opt")]
pub mod merkle_coalesced;

// Main module that re-exports everything for backwards compatibility
pub mod poseidon_interleaved;

#[cfg(feature = "metal-ntt")]
pub mod ntt;
