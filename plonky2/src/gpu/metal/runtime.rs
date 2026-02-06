//! MetalRuntime core struct and buffer allocation methods.
//!
//! Provides the central GPU runtime with pipeline state caching and buffer management.

use lazy_static::lazy_static;
use metal::*;
use std::sync::Mutex;

use crate::gpu::metal::buffer_pool::{BufferPool, BUFFER_POOL};
#[cfg(feature = "metal-bandwidth-opt")]
use crate::gpu::metal::buffer_pool::PersistentMerkleBuffers;
use crate::gpu::metal::tracking::{track_allocation, track_deallocation, TrackedBuffer};

// Shader libraries - include both recursive and linear layouts
const SHADERLIB: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/poseidon_merkle_hasher.metallib"
));

#[cfg(feature = "metal-linear-merkle")]
const SHADERLIB_LINEAR: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/poseidon_merkle_hasher_linear.metallib"
));

#[cfg(feature = "metal-threadgroup")]
const SHADERLIB_THREADGROUP: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/poseidon_merkle_hasher_threadgroup.metallib"
));

// Linear layout + threadgroup memory optimization (combined for metal-optimized)
#[cfg(feature = "metal-optimized")]
const SHADERLIB_LINEAR_THREADGROUP: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/poseidon_merkle_hasher_linear_threadgroup.metallib"
));

// Coalesced memory access for bandwidth optimization (metal-bandwidth-opt)
#[cfg(feature = "metal-bandwidth-opt")]
const SHADERLIB_COALESCED: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/poseidon_merkle_hasher_coalesced.metallib"
));

#[allow(missing_debug_implementations)]
pub struct MetalRuntime {
    pub device: Mutex<Device>,
    // Functions - kept to maintain lifetime; used only during init to create cached PSOs
    #[allow(dead_code)]
    f_poseidon_hash_leaves: Function,
    #[allow(dead_code)]
    f_poseidon_hash_tree_level: Function,
    #[cfg(feature = "metal-linear-merkle")]
    #[allow(dead_code)]
    f_poseidon_hash_leaves_linear: Function,
    #[cfg(feature = "metal-linear-merkle")]
    #[allow(dead_code)]
    f_poseidon_hash_tree_level_linear: Function,
    #[cfg(feature = "metal-linear-merkle")]
    #[allow(dead_code)]
    f_poseidon_hash_caps_linear: Function,
    #[cfg(feature = "metal-threadgroup")]
    #[allow(dead_code)]
    f_poseidon_hash_leaves_threadgroup: Function,
    #[cfg(feature = "metal-threadgroup")]
    #[allow(dead_code)]
    f_poseidon_hash_tree_level_threadgroup: Function,
    #[cfg(feature = "metal-optimized")]
    #[allow(dead_code)]
    f_poseidon_hash_leaves_linear_threadgroup: Function,
    #[cfg(feature = "metal-optimized")]
    #[allow(dead_code)]
    f_poseidon_hash_leaves_and_level1_linear_threadgroup: Function,
    #[cfg(feature = "metal-optimized")]
    #[allow(dead_code)]
    f_poseidon_hash_tree_level_linear_threadgroup: Function,
    #[cfg(feature = "metal-optimized")]
    #[allow(dead_code)]
    f_poseidon_hash_caps_linear_threadgroup: Function,
    // Coalesced shader functions (for metal-bandwidth-opt)
    #[cfg(feature = "metal-bandwidth-opt")]
    #[allow(dead_code)]
    f_poseidon_hash_leaves_coalesced: Function,
    #[cfg(feature = "metal-bandwidth-opt")]
    #[allow(dead_code)]
    f_poseidon_hash_tree_level_coalesced: Function,
    #[cfg(feature = "metal-bandwidth-opt")]
    #[allow(dead_code)]
    f_poseidon_hash_caps_coalesced: Function,
    #[cfg(feature = "metal-bandwidth-opt")]
    #[allow(dead_code)]
    f_poseidon_hash_tree_level_coalesced_instrumented: Function,
    // Cached pipeline states (created once at init, reused for all dispatches)
    pub(crate) pso_poseidon_hash_leaves: ComputePipelineState,
    pub(crate) pso_poseidon_hash_tree_level: ComputePipelineState,
    #[cfg(feature = "metal-linear-merkle")]
    pub(crate) pso_poseidon_hash_leaves_linear: ComputePipelineState,
    #[cfg(feature = "metal-linear-merkle")]
    pub(crate) pso_poseidon_hash_tree_level_linear: ComputePipelineState,
    #[cfg(feature = "metal-linear-merkle")]
    pub(crate) pso_poseidon_hash_caps_linear: ComputePipelineState,
    #[cfg(feature = "metal-threadgroup")]
    pub(crate) pso_poseidon_hash_leaves_threadgroup: ComputePipelineState,
    #[cfg(feature = "metal-threadgroup")]
    pub(crate) pso_poseidon_hash_tree_level_threadgroup: ComputePipelineState,
    #[cfg(feature = "metal-optimized")]
    pub(crate) pso_poseidon_hash_leaves_linear_threadgroup: ComputePipelineState,
    #[cfg(feature = "metal-optimized")]
    pub(crate) pso_poseidon_hash_leaves_and_level1_linear_threadgroup: ComputePipelineState,
    #[cfg(feature = "metal-optimized")]
    pub(crate) pso_poseidon_hash_tree_level_linear_threadgroup: ComputePipelineState,
    #[cfg(feature = "metal-optimized")]
    pub(crate) pso_poseidon_hash_caps_linear_threadgroup: ComputePipelineState,
    // Coalesced pipeline states (for metal-bandwidth-opt)
    #[cfg(feature = "metal-bandwidth-opt")]
    pub(crate) pso_poseidon_hash_leaves_coalesced: ComputePipelineState,
    #[cfg(feature = "metal-bandwidth-opt")]
    pub(crate) pso_poseidon_hash_tree_level_coalesced: ComputePipelineState,
    #[cfg(feature = "metal-bandwidth-opt")]
    pub(crate) pso_poseidon_hash_caps_coalesced: ComputePipelineState,
    #[cfg(feature = "metal-bandwidth-opt")]
    pub(crate) pso_poseidon_hash_tree_level_coalesced_instrumented: ComputePipelineState,
    pub(crate) command_queue: CommandQueue,
    init_time: u128,
    // Persistent pre-allocated buffers for bandwidth optimization (AC-4)
    #[cfg(feature = "metal-bandwidth-opt")]
    pub(crate) persistent_buffers: PersistentMerkleBuffers,
}

unsafe impl Sync for MetalRuntime {}

/// Helper to create a compute pipeline state from a function
fn create_pso(device: &Device, function: &Function) -> ComputePipelineState {
    device
        .new_compute_pipeline_state_with_function(function)
        .unwrap()
}

lazy_static! {
    pub static ref RUNTIME: MetalRuntime = {
        let device = Device::system_default().or_else(|| Device::all().into_iter().next()).expect(
            "No Metal device available. Ensure Metal is supported and no sandbox restrictions block device access.",
        );
        let lib = device.new_library_with_data(SHADERLIB).unwrap();
        let command_queue = device.new_command_queue();

        let f_poseidon_hash_leaves = lib.get_function("poseidon_hash_leaves", None).unwrap();
        let f_poseidon_hash_tree_level = lib.get_function("poseidon_hash_tree_level", None).unwrap();

        // Create cached pipeline states for base functions
        let pso_poseidon_hash_leaves = create_pso(&device, &f_poseidon_hash_leaves);
        let pso_poseidon_hash_tree_level = create_pso(&device, &f_poseidon_hash_tree_level);

        #[cfg(feature = "metal-linear-merkle")]
        let lib_linear = device.new_library_with_data(SHADERLIB_LINEAR).unwrap();
        #[cfg(feature = "metal-linear-merkle")]
        let f_poseidon_hash_leaves_linear = lib_linear.get_function("poseidon_hash_leaves_linear", None).unwrap();
        #[cfg(feature = "metal-linear-merkle")]
        let f_poseidon_hash_tree_level_linear = lib_linear.get_function("poseidon_hash_tree_level_linear", None).unwrap();
        #[cfg(feature = "metal-linear-merkle")]
        let f_poseidon_hash_caps_linear = lib_linear.get_function("poseidon_hash_caps_linear", None).unwrap();
        #[cfg(feature = "metal-linear-merkle")]
        let pso_poseidon_hash_leaves_linear = create_pso(&device, &f_poseidon_hash_leaves_linear);
        #[cfg(feature = "metal-linear-merkle")]
        let pso_poseidon_hash_tree_level_linear = create_pso(&device, &f_poseidon_hash_tree_level_linear);
        #[cfg(feature = "metal-linear-merkle")]
        let pso_poseidon_hash_caps_linear = create_pso(&device, &f_poseidon_hash_caps_linear);

        #[cfg(feature = "metal-threadgroup")]
        let lib_threadgroup = device.new_library_with_data(SHADERLIB_THREADGROUP).unwrap();
        #[cfg(feature = "metal-threadgroup")]
        let f_poseidon_hash_leaves_threadgroup = lib_threadgroup.get_function("poseidon_hash_leaves_threadgroup", None).unwrap();
        #[cfg(feature = "metal-threadgroup")]
        let f_poseidon_hash_tree_level_threadgroup = lib_threadgroup.get_function("poseidon_hash_tree_level_threadgroup", None).unwrap();
        #[cfg(feature = "metal-threadgroup")]
        let pso_poseidon_hash_leaves_threadgroup = create_pso(&device, &f_poseidon_hash_leaves_threadgroup);
        #[cfg(feature = "metal-threadgroup")]
        let pso_poseidon_hash_tree_level_threadgroup = create_pso(&device, &f_poseidon_hash_tree_level_threadgroup);

        #[cfg(feature = "metal-optimized")]
        let lib_linear_threadgroup = device.new_library_with_data(SHADERLIB_LINEAR_THREADGROUP).unwrap();
        #[cfg(feature = "metal-optimized")]
        let f_poseidon_hash_leaves_linear_threadgroup = lib_linear_threadgroup.get_function("poseidon_hash_leaves_linear_threadgroup", None).unwrap();
        #[cfg(feature = "metal-optimized")]
        let f_poseidon_hash_leaves_and_level1_linear_threadgroup = lib_linear_threadgroup.get_function("poseidon_hash_leaves_and_level1_linear_threadgroup", None).unwrap();
        #[cfg(feature = "metal-optimized")]
        let f_poseidon_hash_tree_level_linear_threadgroup = lib_linear_threadgroup.get_function("poseidon_hash_tree_level_linear_threadgroup", None).unwrap();
        #[cfg(feature = "metal-optimized")]
        let f_poseidon_hash_caps_linear_threadgroup = lib_linear_threadgroup.get_function("poseidon_hash_caps_linear_threadgroup", None).unwrap();
        #[cfg(feature = "metal-optimized")]
        let pso_poseidon_hash_leaves_linear_threadgroup = create_pso(&device, &f_poseidon_hash_leaves_linear_threadgroup);
        #[cfg(feature = "metal-optimized")]
        let pso_poseidon_hash_leaves_and_level1_linear_threadgroup = create_pso(&device, &f_poseidon_hash_leaves_and_level1_linear_threadgroup);
        #[cfg(feature = "metal-optimized")]
        let pso_poseidon_hash_tree_level_linear_threadgroup = create_pso(&device, &f_poseidon_hash_tree_level_linear_threadgroup);
        #[cfg(feature = "metal-optimized")]
        let pso_poseidon_hash_caps_linear_threadgroup = create_pso(&device, &f_poseidon_hash_caps_linear_threadgroup);

        // Coalesced shader for bandwidth optimization
        #[cfg(feature = "metal-bandwidth-opt")]
        let lib_coalesced = device.new_library_with_data(SHADERLIB_COALESCED).unwrap();
        #[cfg(feature = "metal-bandwidth-opt")]
        let f_poseidon_hash_leaves_coalesced = lib_coalesced.get_function("poseidon_hash_leaves_coalesced", None).unwrap();
        #[cfg(feature = "metal-bandwidth-opt")]
        let f_poseidon_hash_tree_level_coalesced = lib_coalesced.get_function("poseidon_hash_tree_level_coalesced", None).unwrap();
        #[cfg(feature = "metal-bandwidth-opt")]
        let f_poseidon_hash_caps_coalesced = lib_coalesced.get_function("poseidon_hash_caps_coalesced", None).unwrap();
        #[cfg(feature = "metal-bandwidth-opt")]
        let pso_poseidon_hash_leaves_coalesced = create_pso(&device, &f_poseidon_hash_leaves_coalesced);
        #[cfg(feature = "metal-bandwidth-opt")]
        let pso_poseidon_hash_tree_level_coalesced = create_pso(&device, &f_poseidon_hash_tree_level_coalesced);
        #[cfg(feature = "metal-bandwidth-opt")]
        let pso_poseidon_hash_caps_coalesced = create_pso(&device, &f_poseidon_hash_caps_coalesced);
        #[cfg(feature = "metal-bandwidth-opt")]
        let f_poseidon_hash_tree_level_coalesced_instrumented = lib_coalesced.get_function("poseidon_hash_tree_level_coalesced_instrumented", None).unwrap();
        #[cfg(feature = "metal-bandwidth-opt")]
        let pso_poseidon_hash_tree_level_coalesced_instrumented = create_pso(&device, &f_poseidon_hash_tree_level_coalesced_instrumented);

        MetalRuntime {
            device: Mutex::new(Device::system_default().unwrap()),
            command_queue,
            f_poseidon_hash_leaves,
            f_poseidon_hash_tree_level,
            pso_poseidon_hash_leaves,
            pso_poseidon_hash_tree_level,
            #[cfg(feature = "metal-linear-merkle")]
            f_poseidon_hash_leaves_linear,
            #[cfg(feature = "metal-linear-merkle")]
            f_poseidon_hash_tree_level_linear,
            #[cfg(feature = "metal-linear-merkle")]
            f_poseidon_hash_caps_linear,
            #[cfg(feature = "metal-linear-merkle")]
            pso_poseidon_hash_leaves_linear,
            #[cfg(feature = "metal-linear-merkle")]
            pso_poseidon_hash_tree_level_linear,
            #[cfg(feature = "metal-linear-merkle")]
            pso_poseidon_hash_caps_linear,
            #[cfg(feature = "metal-threadgroup")]
            f_poseidon_hash_leaves_threadgroup,
            #[cfg(feature = "metal-threadgroup")]
            f_poseidon_hash_tree_level_threadgroup,
            #[cfg(feature = "metal-threadgroup")]
            pso_poseidon_hash_leaves_threadgroup,
            #[cfg(feature = "metal-threadgroup")]
            pso_poseidon_hash_tree_level_threadgroup,
            #[cfg(feature = "metal-optimized")]
            f_poseidon_hash_leaves_linear_threadgroup,
            #[cfg(feature = "metal-optimized")]
            f_poseidon_hash_leaves_and_level1_linear_threadgroup,
            #[cfg(feature = "metal-optimized")]
            f_poseidon_hash_tree_level_linear_threadgroup,
            #[cfg(feature = "metal-optimized")]
            f_poseidon_hash_caps_linear_threadgroup,
            #[cfg(feature = "metal-optimized")]
            pso_poseidon_hash_leaves_linear_threadgroup,
            #[cfg(feature = "metal-optimized")]
            pso_poseidon_hash_leaves_and_level1_linear_threadgroup,
            #[cfg(feature = "metal-optimized")]
            pso_poseidon_hash_tree_level_linear_threadgroup,
            #[cfg(feature = "metal-optimized")]
            pso_poseidon_hash_caps_linear_threadgroup,
            #[cfg(feature = "metal-bandwidth-opt")]
            f_poseidon_hash_leaves_coalesced,
            #[cfg(feature = "metal-bandwidth-opt")]
            f_poseidon_hash_tree_level_coalesced,
            #[cfg(feature = "metal-bandwidth-opt")]
            f_poseidon_hash_caps_coalesced,
            #[cfg(feature = "metal-bandwidth-opt")]
            pso_poseidon_hash_leaves_coalesced,
            #[cfg(feature = "metal-bandwidth-opt")]
            pso_poseidon_hash_tree_level_coalesced,
            #[cfg(feature = "metal-bandwidth-opt")]
            pso_poseidon_hash_caps_coalesced,
            #[cfg(feature = "metal-bandwidth-opt")]
            f_poseidon_hash_tree_level_coalesced_instrumented,
            #[cfg(feature = "metal-bandwidth-opt")]
            pso_poseidon_hash_tree_level_coalesced_instrumented,
            // Initialize persistent buffers for bandwidth optimization (AC-4)
            #[cfg(feature = "metal-bandwidth-opt")]
            persistent_buffers: {
                use std::sync::Arc;
                // Pre-allocate for degree 2^22 by default (practical for tests)
                // Can be expanded via runtime API if needed for larger trees
                let arc_device = Arc::new(Device::system_default().unwrap());
                PersistentMerkleBuffers::new(arc_device, 22)
            },
            init_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros(),
        }
    };
}

impl MetalRuntime {
    pub fn get() -> &'static MetalRuntime {
        &RUNTIME
    }

    pub fn warm_up(&self) -> u128 {
        self.init_time
    }

    /// Get reference to persistent buffers for bandwidth optimization (AC-4)
    #[cfg(feature = "metal-bandwidth-opt")]
    pub fn get_persistent_buffers(&self) -> &PersistentMerkleBuffers {
        &self.persistent_buffers
    }

    /// Check if persistent buffers can accommodate the given leaf count
    #[cfg(feature = "metal-bandwidth-opt")]
    pub fn persistent_buffers_fit(&self, leaf_count: usize) -> bool {
        self.persistent_buffers.fits(leaf_count)
    }

    /// Get the buffer addresses from persistent buffers for verification
    #[cfg(feature = "metal-bandwidth-opt")]
    pub fn get_persistent_buffer_addresses(&self) -> (usize, usize) {
        self.persistent_buffers.buffer_addresses()
    }

    /// Get cached pipeline state for tree level hashing (no allocation on each call)
    pub fn get_poseidon_hash_tree_level_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_tree_level
    }

    /// Get cached pipeline state for leaf hashing (no allocation on each call)
    pub fn get_poseidon_hash_leaves_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_leaves
    }

    pub fn alloc(&self, len: usize) -> Buffer {
        track_allocation(len);
        self.device
            .lock()
            .unwrap()
            .new_buffer(len as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Allocate buffer with 256-byte alignment for optimal memory coalescing
    /// Metal GPUs benefit from aligned buffer sizes for memory access patterns
    /// Uses buffer pooling to reuse previously allocated buffers
    pub fn alloc_aligned(&self, len: usize) -> Buffer {
        const ALIGNMENT: usize = 256;
        let aligned_len = (len + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        // Try to get a buffer from the pool first
        if let Some(buffer) = BUFFER_POOL.get(aligned_len) {
            track_allocation(buffer.length() as usize);
            return buffer;
        }

        // Allocate new buffer at size class (power of 2) for better pool reuse
        let size_class = BufferPool::size_class(aligned_len);
        track_allocation(size_class);
        self.device
            .lock()
            .unwrap()
            .new_buffer(size_class as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Allocate buffer with 256-byte alignment, returning whether it came from pool
    /// Buffers from pool don't need zeroing since GPU writes all positions
    pub fn alloc_aligned_with_pool_hint(&self, len: usize) -> (Buffer, bool) {
        const ALIGNMENT: usize = 256;
        let aligned_len = (len + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        // Try to get a buffer from the pool first
        if let Some(buffer) = BUFFER_POOL.get(aligned_len) {
            track_allocation(buffer.length() as usize);
            return (buffer, true); // from_pool = true
        }

        // Allocate new buffer at size class (power of 2) for better pool reuse
        let size_class = BufferPool::size_class(aligned_len);
        track_allocation(size_class);
        let buffer = self.device
            .lock()
            .unwrap()
            .new_buffer(size_class as u64, MTLResourceOptions::StorageModeShared);
        (buffer, false) // from_pool = false
    }

    /// Return a buffer to the pool for reuse
    /// Call this when done with a buffer instead of letting it drop
    pub fn return_buffer(&self, buffer: Buffer) {
        // Track deallocation (buffer no longer in use)
        track_deallocation(buffer.length() as usize);
        BUFFER_POOL.put(buffer);
    }

    /// Allocate a tracked buffer (auto-decrements on drop)
    pub fn alloc_tracked(&self, len: usize) -> TrackedBuffer {
        let buffer = self.device
            .lock()
            .unwrap()
            .new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
        TrackedBuffer::new(buffer)
    }

    /// Allocate 256-byte aligned buffer with RAII tracking (auto-decrements on drop)
    /// Uses buffer pooling internally; pool buffers are tracked on retrieval
    pub fn alloc_aligned_tracked(&self, len: usize) -> TrackedBuffer {
        const ALIGNMENT: usize = 256;
        let aligned_len = (len + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        // Try to get a buffer from the pool first
        if let Some(buffer) = BUFFER_POOL.get(aligned_len) {
            return TrackedBuffer::from_pool(buffer);
        }

        // Allocate new buffer at size class (power of 2) for better pool reuse
        let size_class = BufferPool::size_class(aligned_len);
        let buffer = self.device
            .lock()
            .unwrap()
            .new_buffer(size_class as u64, MTLResourceOptions::StorageModeShared);
        TrackedBuffer::new(buffer)
    }

    /// Allocate 256-byte aligned buffer with RAII tracking, returning whether from pool
    pub fn alloc_aligned_tracked_with_pool_hint(&self, len: usize) -> (TrackedBuffer, bool) {
        const ALIGNMENT: usize = 256;
        let aligned_len = (len + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        // Try to get a buffer from the pool first
        if let Some(buffer) = BUFFER_POOL.get(aligned_len) {
            return (TrackedBuffer::from_pool(buffer), true);
        }

        // Allocate new buffer at size class (power of 2) for better pool reuse
        let size_class = BufferPool::size_class(aligned_len);
        let buffer = self.device
            .lock()
            .unwrap()
            .new_buffer(size_class as u64, MTLResourceOptions::StorageModeShared);
        (TrackedBuffer::new(buffer), false)
    }

    /// Return a TrackedBuffer to the pool for reuse
    /// Tracks deallocation since buffer is no longer "in use" (pool holds it for future reuse)
    pub fn return_tracked_buffer(&self, tracked: TrackedBuffer) {
        // Use into_inner() which calls track_deallocation - buffer is no longer in active use
        let buffer = tracked.into_inner();
        BUFFER_POOL.put(buffer);
    }

    /// Allocate a buffer with data, tracking its size
    /// This tracks the allocation; caller is responsible for ensuring
    /// track_deallocation is called when buffer is no longer needed
    pub fn alloc_with_data<T>(&self, data: &[T]) -> Buffer {
        let len = std::mem::size_of_val(data);
        track_allocation(len);
        self.device
            .lock()
            .unwrap()
            .new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                len as u64,
                MTLResourceOptions::StorageModeShared,
            )
    }

    /// Allocate a tracked buffer with data (auto-decrements on drop)
    pub fn alloc_with_data_tracked<T>(&self, data: &[T]) -> TrackedBuffer {
        let len = std::mem::size_of_val(data);
        let buffer = self.device
            .lock()
            .unwrap()
            .new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                len as u64,
                MTLResourceOptions::StorageModeShared,
            );
        TrackedBuffer::new(buffer)
    }

    /// Round size up to 256-byte alignment
    pub const fn align_to_256(len: usize) -> usize {
        const ALIGNMENT: usize = 256;
        (len + ALIGNMENT - 1) & !(ALIGNMENT - 1)
    }

    #[cfg(feature = "metal-linear-merkle")]
    pub fn get_poseidon_hash_leaves_linear_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_leaves_linear
    }

    #[cfg(feature = "metal-linear-merkle")]
    pub fn get_poseidon_hash_tree_level_linear_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_tree_level_linear
    }

    #[cfg(feature = "metal-linear-merkle")]
    pub fn get_poseidon_hash_caps_linear_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_caps_linear
    }

    #[cfg(feature = "metal-threadgroup")]
    pub fn get_poseidon_hash_leaves_threadgroup_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_leaves_threadgroup
    }

    #[cfg(feature = "metal-threadgroup")]
    pub fn get_poseidon_hash_tree_level_threadgroup_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_tree_level_threadgroup
    }

    #[cfg(feature = "metal-optimized")]
    pub fn get_poseidon_hash_leaves_linear_threadgroup_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_leaves_linear_threadgroup
    }

    #[cfg(feature = "metal-optimized")]
    pub fn get_poseidon_hash_leaves_and_level1_linear_threadgroup_pipeline_state(
        &self,
    ) -> &ComputePipelineState {
        &self.pso_poseidon_hash_leaves_and_level1_linear_threadgroup
    }

    #[cfg(feature = "metal-optimized")]
    pub fn get_poseidon_hash_tree_level_linear_threadgroup_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_tree_level_linear_threadgroup
    }

    #[cfg(feature = "metal-optimized")]
    pub fn get_poseidon_hash_caps_linear_threadgroup_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_caps_linear_threadgroup
    }

    #[cfg(feature = "metal-bandwidth-opt")]
    pub fn get_poseidon_hash_leaves_coalesced_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_leaves_coalesced
    }

    #[cfg(feature = "metal-bandwidth-opt")]
    pub fn get_poseidon_hash_tree_level_coalesced_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_tree_level_coalesced
    }

    #[cfg(feature = "metal-bandwidth-opt")]
    pub fn get_poseidon_hash_caps_coalesced_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_caps_coalesced
    }

    #[cfg(feature = "metal-bandwidth-opt")]
    pub fn get_poseidon_hash_tree_level_coalesced_instrumented_pipeline_state(&self) -> &ComputePipelineState {
        &self.pso_poseidon_hash_tree_level_coalesced_instrumented
    }
}
