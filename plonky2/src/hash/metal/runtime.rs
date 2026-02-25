//! MetalRuntime core struct and buffer allocation methods.
//!
//! Provides the central GPU runtime with pipeline state caching and buffer management.
//! Only the linear+threadgroup Poseidon Merkle shader is loaded.

use metal::*;
use once_cell::sync::Lazy;
use std::sync::Mutex;

use crate::hash::metal::buffer_pool::{BufferPool, BUFFER_POOL};
use crate::hash::metal::tracking::{track_allocation, track_deallocation, TrackedBuffer};

/// Pre-compiled linear+threadgroup shader library embedded at build time.
const SHADERLIB_LINEAR_THREADGROUP: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/poseidon_merkle_hasher_linear_threadgroup.metallib"
));

pub struct MetalRuntime {
    pub device: Mutex<Device>,
    // Functions — kept to maintain lifetime; only used during init to create cached PSOs
    #[allow(dead_code)]
    f_poseidon_hash_leaves_linear_threadgroup: Function,
    #[allow(dead_code)]
    f_poseidon_hash_tree_level_linear_threadgroup: Function,
    #[allow(dead_code)]
    f_poseidon_hash_caps_linear_threadgroup: Function,
    // Cached pipeline states (created once at init, reused for all dispatches)
    pub(crate) pso_poseidon_hash_leaves_linear_threadgroup: ComputePipelineState,
    pub(crate) pso_poseidon_hash_tree_level_linear_threadgroup: ComputePipelineState,
    pub(crate) pso_poseidon_hash_caps_linear_threadgroup: ComputePipelineState,
    pub(crate) command_queue: CommandQueue,
    init_time: u128,
}

unsafe impl Sync for MetalRuntime {}

/// Helper to create a compute pipeline state from a function.
fn create_pso(device: &Device, function: &Function) -> ComputePipelineState {
    device
        .new_compute_pipeline_state_with_function(function)
        .unwrap()
}

pub static RUNTIME: Lazy<MetalRuntime> = Lazy::new(|| {
    let device = Device::system_default().unwrap();
    let lib_lt = device
        .new_library_with_data(SHADERLIB_LINEAR_THREADGROUP)
        .unwrap();
    let command_queue = device.new_command_queue();

    let f_leaves = lib_lt
        .get_function("poseidon_hash_leaves_linear_threadgroup", None)
        .unwrap();
    let f_level = lib_lt
        .get_function("poseidon_hash_tree_level_linear_threadgroup", None)
        .unwrap();
    let f_caps = lib_lt
        .get_function("poseidon_hash_caps_linear_threadgroup", None)
        .unwrap();

    let pso_leaves = create_pso(&device, &f_leaves);
    let pso_level = create_pso(&device, &f_level);
    let pso_caps = create_pso(&device, &f_caps);

    MetalRuntime {
        device: Mutex::new(Device::system_default().unwrap()),
        command_queue,
        pso_poseidon_hash_leaves_linear_threadgroup: pso_leaves,
        pso_poseidon_hash_tree_level_linear_threadgroup: pso_level,
        pso_poseidon_hash_caps_linear_threadgroup: pso_caps,
        f_poseidon_hash_leaves_linear_threadgroup: f_leaves,
        f_poseidon_hash_tree_level_linear_threadgroup: f_level,
        f_poseidon_hash_caps_linear_threadgroup: f_caps,
        init_time: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros(),
    }
});

impl MetalRuntime {
    pub fn get() -> &'static MetalRuntime {
        &RUNTIME
    }

    pub fn warm_up(&self) -> u128 {
        self.init_time
    }

    // ── Pipeline state accessors ──────────────────────────────────────────────

    pub fn get_poseidon_hash_leaves_linear_threadgroup_pipeline_state(
        &self,
    ) -> &ComputePipelineState {
        &self.pso_poseidon_hash_leaves_linear_threadgroup
    }

    pub fn get_poseidon_hash_tree_level_linear_threadgroup_pipeline_state(
        &self,
    ) -> &ComputePipelineState {
        &self.pso_poseidon_hash_tree_level_linear_threadgroup
    }

    pub fn get_poseidon_hash_caps_linear_threadgroup_pipeline_state(
        &self,
    ) -> &ComputePipelineState {
        &self.pso_poseidon_hash_caps_linear_threadgroup
    }

    // ── Buffer allocation ─────────────────────────────────────────────────────

    pub fn alloc(&self, len: usize) -> Buffer {
        track_allocation(len);
        self.device
            .lock()
            .unwrap()
            .new_buffer(len as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Allocate buffer with 256-byte alignment for optimal memory coalescing.
    /// Uses buffer pooling to reuse previously allocated buffers.
    pub fn alloc_aligned(&self, len: usize) -> Buffer {
        const ALIGNMENT: usize = 256;
        let aligned_len = (len + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        if let Some(buffer) = BUFFER_POOL.get(aligned_len) {
            track_allocation(buffer.length() as usize);
            return buffer;
        }

        let size_class = BufferPool::size_class(aligned_len);
        track_allocation(size_class);
        self.device
            .lock()
            .unwrap()
            .new_buffer(size_class as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Allocate 256-byte aligned buffer, returning whether it came from pool.
    pub fn alloc_aligned_with_pool_hint(&self, len: usize) -> (Buffer, bool) {
        const ALIGNMENT: usize = 256;
        let aligned_len = (len + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        if let Some(buffer) = BUFFER_POOL.get(aligned_len) {
            track_allocation(buffer.length() as usize);
            return (buffer, true);
        }

        let size_class = BufferPool::size_class(aligned_len);
        track_allocation(size_class);
        let buffer = self
            .device
            .lock()
            .unwrap()
            .new_buffer(size_class as u64, MTLResourceOptions::StorageModeShared);
        (buffer, false)
    }

    pub fn alloc_tracked(&self, len: usize) -> TrackedBuffer {
        let buffer = self
            .device
            .lock()
            .unwrap()
            .new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
        TrackedBuffer::new(buffer)
    }

    /// Allocate 256-byte aligned buffer with RAII tracking.
    pub fn alloc_aligned_tracked(&self, len: usize) -> TrackedBuffer {
        const ALIGNMENT: usize = 256;
        let aligned_len = (len + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        if let Some(buffer) = BUFFER_POOL.get(aligned_len) {
            return TrackedBuffer::from_pool(buffer);
        }

        let size_class = BufferPool::size_class(aligned_len);
        let buffer = self
            .device
            .lock()
            .unwrap()
            .new_buffer(size_class as u64, MTLResourceOptions::StorageModeShared);
        TrackedBuffer::new(buffer)
    }

    /// Allocate 256-byte aligned buffer with RAII tracking, returning whether from pool.
    pub fn alloc_aligned_tracked_with_pool_hint(&self, len: usize) -> (TrackedBuffer, bool) {
        const ALIGNMENT: usize = 256;
        let aligned_len = (len + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        if let Some(buffer) = BUFFER_POOL.get(aligned_len) {
            return (TrackedBuffer::from_pool(buffer), true);
        }

        let size_class = BufferPool::size_class(aligned_len);
        let buffer = self
            .device
            .lock()
            .unwrap()
            .new_buffer(size_class as u64, MTLResourceOptions::StorageModeShared);
        (TrackedBuffer::new(buffer), false)
    }

    /// Return a buffer to the pool for reuse.
    pub fn return_buffer(&self, buffer: Buffer) {
        track_deallocation(buffer.length() as usize);
        BUFFER_POOL.put(buffer);
    }

    /// Return a TrackedBuffer to the pool for reuse.
    pub fn return_tracked_buffer(&self, tracked: TrackedBuffer) {
        let buffer = tracked.into_inner();
        BUFFER_POOL.put(buffer);
    }

    /// Allocate a buffer with data, tracking its size.
    /// Caller is responsible for calling `track_deallocation` when done.
    pub fn alloc_with_data<T>(&self, data: &[T]) -> Buffer {
        let len = std::mem::size_of_val(data);
        track_allocation(len);
        self.device.lock().unwrap().new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Allocate a tracked buffer with data (auto-decrements on drop).
    pub fn alloc_with_data_tracked<T>(&self, data: &[T]) -> TrackedBuffer {
        let len = std::mem::size_of_val(data);
        let buffer = self.device.lock().unwrap().new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        TrackedBuffer::new(buffer)
    }

    /// Round size up to 256-byte alignment.
    pub const fn align_to_256(len: usize) -> usize {
        const ALIGNMENT: usize = 256;
        (len + ALIGNMENT - 1) & !(ALIGNMENT - 1)
    }
}
