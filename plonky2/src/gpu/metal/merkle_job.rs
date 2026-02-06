//! Async Merkle tree GPU job handling.
//!
//! MerkleGpuJob allows submitting GPU work and doing CPU work while GPU is busy.

use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::*;
use crate::{field::goldilocks_field::GoldilocksField, hash::hash_types::HashOut};

use crate::gpu::metal::tracking::track_deallocation;
use crate::gpu::metal::utils::{convert_linear_to_recursive, from_buf_raw};

// External C function for retaining Objective-C objects (for async command buffer handling)
extern "C" {
    fn objc_retain(obj: *mut metal::objc::runtime::Object) -> *mut metal::objc::runtime::Object;
}

/// Handle for an in-flight async Merkle tree build operation
/// Allows submitting GPU work and doing CPU work while GPU is busy
pub struct MerkleGpuJob {
    /// Owned CommandBuffer (retained)
    command_buffer: CommandBuffer,
    /// Output buffer containing tree digests
    digests_buffer: Buffer,
    /// Size of digests buffer for deallocation tracking
    digests_buffer_size: usize,
    /// Optional caps buffer (used by linear layouts that compute caps separately)
    caps_buffer: Option<Buffer>,
    /// Size of caps buffer for deallocation tracking
    caps_buffer_size: usize,
    /// Total number of digests in the output
    total_digests: usize,
    /// Number of caps (2^cap_height)
    num_caps: usize,
    /// Tree height
    tree_height: usize,
    /// Cap height
    cap_height: usize,
    /// Whether this uses linear layout (needs conversion to recursive on finish)
    is_linear_layout: bool,
}

impl std::fmt::Debug for MerkleGpuJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MerkleGpuJob")
            .field("total_digests", &self.total_digests)
            .field("num_caps", &self.num_caps)
            .field("tree_height", &self.tree_height)
            .field("cap_height", &self.cap_height)
            .field("is_linear_layout", &self.is_linear_layout)
            .finish_non_exhaustive()
    }
}

// Safety: MerkleGpuJob only accesses Metal objects which are thread-safe
unsafe impl Send for MerkleGpuJob {}
unsafe impl Sync for MerkleGpuJob {}

/// Result of async Merkle tree build
#[allow(missing_debug_implementations)]
pub struct MerkleGpuResult {
    /// Tree digests in plonky2 recursive format
    pub digests: Vec<HashOut<GoldilocksField>>,
    /// Cap hashes (optional - some implementations compute these separately)
    pub caps: Option<Vec<HashOut<GoldilocksField>>>,
}

impl MerkleGpuJob {
    /// Create a new job by retaining the command buffer
    pub unsafe fn new(
        command_buffer: &CommandBufferRef,
        digests_buffer: Buffer,
        caps_buffer: Option<Buffer>,
        total_digests: usize,
        num_caps: usize,
        tree_height: usize,
        cap_height: usize,
        is_linear_layout: bool,
    ) -> Self {
        let ptr = command_buffer.as_ptr() as *mut metal::objc::runtime::Object;
        let retained_ptr = objc_retain(ptr) as *mut MTLCommandBuffer;
        let owned = CommandBuffer::from_ptr(retained_ptr);

        // Store buffer sizes for deallocation tracking in finish()
        let digests_buffer_size = digests_buffer.length() as usize;
        let caps_buffer_size = caps_buffer.as_ref().map(|b| b.length() as usize).unwrap_or(0);

        Self {
            command_buffer: owned,
            digests_buffer,
            digests_buffer_size,
            caps_buffer,
            caps_buffer_size,
            total_digests,
            num_caps,
            tree_height,
            cap_height,
            is_linear_layout,
        }
    }

    /// Check if the GPU operation is complete without blocking
    pub fn is_complete(&self) -> bool {
        self.command_buffer.status() == MTLCommandBufferStatus::Completed
    }

    /// Wait for the GPU operation to complete and retrieve digests and caps
    /// For linear layout, this performs the linear→recursive conversion
    pub fn finish(mut self) -> MerkleGpuResult {
        self.command_buffer.wait_until_completed();

        let ptr = self.digests_buffer.contents() as *mut HashOut<GoldilocksField>;
        let raw_digests = unsafe { from_buf_raw::<HashOut<GoldilocksField>>(ptr, self.total_digests) };

        let digests = if self.is_linear_layout {
            // Convert from linear to recursive layout
            let num_layers = self.tree_height - self.cap_height;
            let tree_length = self.total_digests >> self.cap_height;
            convert_linear_to_recursive(&raw_digests, num_layers, tree_length, self.num_caps)
        } else {
            raw_digests
        };

        let caps = self.caps_buffer.take().map(|buf| {
            let caps_ptr = buf.contents() as *mut HashOut<GoldilocksField>;
            unsafe { from_buf_raw::<HashOut<GoldilocksField>>(caps_ptr, self.num_caps) }
        });

        // Track deallocation of buffers and zero sizes to prevent Drop from double-tracking
        track_deallocation(self.digests_buffer_size);
        self.digests_buffer_size = 0;
        if self.caps_buffer_size > 0 {
            track_deallocation(self.caps_buffer_size);
            self.caps_buffer_size = 0;
        }

        MerkleGpuResult { digests, caps }
    }

    /// Get tree metadata
    pub fn tree_height(&self) -> usize {
        self.tree_height
    }

    pub fn cap_height(&self) -> usize {
        self.cap_height
    }
}

/// Safety net: if MerkleGpuJob is dropped without calling finish(), track deallocation
impl Drop for MerkleGpuJob {
    fn drop(&mut self) {
        // Only track deallocation if sizes are non-zero (not already consumed by finish())
        // Note: finish() takes self by value, but the sizes remain in the moved struct
        // This handles the case where the job is dropped without calling finish()
        if self.digests_buffer_size > 0 {
            track_deallocation(self.digests_buffer_size);
            self.digests_buffer_size = 0; // Prevent double-tracking
        }
        if self.caps_buffer_size > 0 {
            track_deallocation(self.caps_buffer_size);
            self.caps_buffer_size = 0;
        }
    }
}
