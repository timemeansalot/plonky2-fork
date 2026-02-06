//! Metal GPU acceleration for Poseidon hashing and Merkle tree construction.
//!
//! This module provides GPU-accelerated implementations of:
//! - Poseidon hash function for Goldilocks field
//! - Merkle tree construction with various layout optimizations
//!
//! The implementation is split into several submodules:
//! - `tracking`: Buffer allocation tracking and RAII wrappers
//! - `buffer_pool`: Buffer pooling for efficient reuse
//! - `merkle_job`: Async GPU job handling
//! - `threadgroup_config`: Threadgroup size configuration
//! - `utils`: Utility functions and layout conversion
//! - `runtime`: Core MetalRuntime struct and allocation methods
//! - `merkle_recursive`: Base recursive layout implementation
//! - `merkle_threadgroup`: Threadgroup-optimized implementation
//! - `merkle_linear`: Linear layout implementation
//! - `merkle_linear_threadgroup`: Combined linear + threadgroup implementation

// Re-export all public items from submodules for backwards compatibility

// Tracking module
pub use crate::gpu::metal::tracking::{
    get_allocation_count,
    get_allocation_stats,
    get_allocation_stats_string,
    get_current_allocated_bytes,
    get_peak_allocated_bytes,
    reset_allocation_stats,
    track_allocation,
    track_deallocation,
    TrackedBuffer,
};

// Buffer pool module
pub use crate::gpu::metal::buffer_pool::{
    clear_buffer_pool,
    get_buffer_pool_stats,
    reset_buffer_pool_stats,
    BufferPool,
    PersistentMerkleBuffers,
    BUFFER_POOL,
};

// Merkle job module
pub use crate::gpu::metal::merkle_job::{MerkleGpuJob, MerkleGpuResult};

// Threadgroup config module
pub use crate::gpu::metal::threadgroup_config::{
    compute_effective_threadgroup_size,
    get_merkle_threadgroup_config,
    get_merkle_threadgroup_size,
};

// Utils module
pub use crate::gpu::metal::utils::{
    convert_linear_to_recursive,
    from_buf_raw,
    get_node_hash_index_in_digests,
    get_size_for_count,
    AccessCounters,
    CoalescedUniforms,
    LinearUniforms,
    POSEIDON_CONST_SIZE,
    POSEIDON_MDS_SIZE,
    POSEIDON_RC_SIZE,
};

// Runtime module
pub use crate::gpu::metal::runtime::MetalRuntime;

// Coalesced/chunked module (bandwidth optimization)
#[cfg(feature = "metal-bandwidth-opt")]
pub use crate::gpu::metal::merkle_coalesced::{
    get_chunk_size_log2,
    set_chunk_size_log2,
    is_double_buffer_enabled,
    set_double_buffer_enabled,
    get_coalesced_height_threshold,
    set_coalesced_height_threshold,
};

// RAII guards for test serialization
#[cfg(feature = "metal-bandwidth-opt")]
pub use crate::gpu::metal::merkle_coalesced::{
    CoalescedThresholdGuard,
    ChunkSizeGuard,
    DoubleBufferGuard,
};

// TRUE chunked processing with root merge (AC-2)
#[cfg(feature = "metal-bandwidth-opt")]
pub use crate::gpu::metal::merkle_coalesced::{
    get_last_chunk_dispatch_count,
    reset_chunk_dispatch_count,
};

// Sublevel chunking counters (AC-1)
#[cfg(feature = "metal-bandwidth-opt")]
pub use crate::gpu::metal::merkle_coalesced::{
    get_last_level_dispatch_count,
    reset_level_dispatch_count,
};

// Staging buffer counters (AC-3)
#[cfg(feature = "metal-bandwidth-opt")]
pub use crate::gpu::metal::merkle_coalesced::{
    get_staging_buffer_switches,
    reset_staging_buffer_switches,
};
