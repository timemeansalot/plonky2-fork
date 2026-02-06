//! Buffer pool for reusing Metal buffers to avoid allocation overhead.
//! Groups buffers by size class (power of 2) for efficient reuse.

use lazy_static::lazy_static;
use metal::Buffer;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Buffer pool for reusing Metal buffers to avoid allocation overhead
/// Groups buffers by size class (power of 2) for efficient reuse
#[allow(missing_debug_implementations)]
pub struct BufferPool {
    /// Map from buffer size to list of available buffers
    pools: Mutex<HashMap<usize, Vec<Buffer>>>,
    /// Total bytes currently in pool (for monitoring)
    pool_bytes: AtomicUsize,
    /// Number of pool hits (buffer reused)
    hits: AtomicUsize,
    /// Number of pool misses (new allocation needed)
    misses: AtomicUsize,
}

impl BufferPool {
    pub fn new() -> Self {
        BufferPool {
            pools: Mutex::new(HashMap::new()),
            pool_bytes: AtomicUsize::new(0),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Round size up to next size class (power of 2, minimum 4KB)
    pub fn size_class(size: usize) -> usize {
        const MIN_SIZE: usize = 4096;
        let size = size.max(MIN_SIZE);
        size.next_power_of_two()
    }

    /// Try to get a buffer from the pool
    pub fn get(&self, size: usize) -> Option<Buffer> {
        let size_class = Self::size_class(size);
        let mut pools = self.pools.lock().unwrap();
        if let Some(buffers) = pools.get_mut(&size_class) {
            if let Some(buffer) = buffers.pop() {
                self.pool_bytes.fetch_sub(size_class, Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(buffer);
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Return a buffer to the pool for reuse
    pub fn put(&self, buffer: Buffer) {
        let size = buffer.length() as usize;
        let size_class = Self::size_class(size);

        // Limit pool size to prevent unbounded memory growth (max 2GB)
        const MAX_POOL_BYTES: usize = 2 * 1024 * 1024 * 1024;
        if self.pool_bytes.load(Ordering::Relaxed) + size_class > MAX_POOL_BYTES {
            // Drop buffer instead of pooling
            return;
        }

        let mut pools = self.pools.lock().unwrap();
        pools.entry(size_class).or_insert_with(Vec::new).push(buffer);
        self.pool_bytes.fetch_add(size_class, Ordering::Relaxed);
    }

    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.pool_bytes.load(Ordering::Relaxed),
        )
    }

    /// Clear all pooled buffers
    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        pools.clear();
        self.pool_bytes.store(0, Ordering::Relaxed);
    }

    /// Reset statistics counters (for per-degree measurement)
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

lazy_static! {
    pub static ref BUFFER_POOL: BufferPool = BufferPool::new();
}

/// Get buffer pool statistics (hits, misses, bytes_pooled)
pub fn get_buffer_pool_stats() -> (usize, usize, usize) {
    BUFFER_POOL.stats()
}

/// Reset buffer pool statistics (for per-degree measurement)
pub fn reset_buffer_pool_stats() {
    BUFFER_POOL.reset_stats();
}

/// Clear the buffer pool
pub fn clear_buffer_pool() {
    BUFFER_POOL.clear();
}

// ========== Persistent Pre-allocated Buffers ==========

use metal::{Device, MTLResourceOptions};
use std::sync::Arc;

/// Default maximum degree for pre-allocation (2^25)
const DEFAULT_MAX_DEGREE_LOG2: usize = 25;

/// Persistent pre-allocated buffers for Merkle tree construction.
/// Pre-allocates buffers for a maximum tree size at startup to eliminate
/// allocation overhead during tree building.
#[allow(missing_debug_implementations)]
pub struct PersistentMerkleBuffers {
    /// Pre-allocated digests buffer
    digests_buffer: Buffer,
    /// Pre-allocated caps buffer
    caps_buffer: Buffer,
    /// Maximum number of leaves this can handle
    max_leaves: usize,
    /// Maximum degree (log2) this was sized for
    max_degree_log2: usize,
    /// Device reference for fallback allocation
    #[allow(dead_code)]
    device: Arc<Device>,
}

impl PersistentMerkleBuffers {
    /// Create persistent buffers pre-allocated for the given maximum degree.
    ///
    /// # Arguments
    /// * `device` - Metal device for buffer allocation
    /// * `max_degree_log2` - Maximum tree height to support (default: 25)
    ///
    /// # Panics
    /// Panics if max_degree_log2 > 26 (too large for practical use)
    pub fn new(device: Arc<Device>, max_degree_log2: usize) -> Self {
        assert!(
            max_degree_log2 <= 26,
            "Maximum degree {} too large (max 26)", max_degree_log2
        );

        let max_leaves = 1usize << max_degree_log2;

        // Calculate buffer sizes for worst-case (cap_height = 0)
        // Digests: 2 * max_leaves - 1 nodes, each 4 u64s (32 bytes)
        let max_digests = 2 * max_leaves;
        let digests_size = max_digests * 4 * std::mem::size_of::<u64>();

        // Caps: up to max_leaves caps (worst case cap_height = tree_height)
        let caps_size = max_leaves * 4 * std::mem::size_of::<u64>();

        println!(
            "[PersistentBuffers] Pre-allocating for degree 2^{}: digests={}MB, caps={}MB",
            max_degree_log2,
            digests_size / (1024 * 1024),
            caps_size / (1024 * 1024)
        );

        let digests_buffer = device.new_buffer(
            digests_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let caps_buffer = device.new_buffer(
            caps_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Zero-initialize buffers
        unsafe {
            std::ptr::write_bytes(digests_buffer.contents() as *mut u8, 0, digests_size);
            std::ptr::write_bytes(caps_buffer.contents() as *mut u8, 0, caps_size);
        }

        Self {
            digests_buffer,
            caps_buffer,
            max_leaves,
            max_degree_log2,
            device,
        }
    }

    /// Create with default maximum degree (2^25)
    pub fn with_default_size(device: Arc<Device>) -> Self {
        Self::new(device, DEFAULT_MAX_DEGREE_LOG2)
    }

    /// Check if the requested tree size fits in pre-allocated buffers
    pub fn fits(&self, leaf_count: usize) -> bool {
        leaf_count <= self.max_leaves
    }

    /// Get the maximum supported leaf count
    pub fn max_leaves(&self) -> usize {
        self.max_leaves
    }

    /// Get the maximum supported degree (log2)
    pub fn max_degree_log2(&self) -> usize {
        self.max_degree_log2
    }

    /// Get reference to pre-allocated digests buffer
    pub fn digests_buffer(&self) -> &Buffer {
        &self.digests_buffer
    }

    /// Get reference to pre-allocated caps buffer
    pub fn caps_buffer(&self) -> &Buffer {
        &self.caps_buffer
    }

    /// Get the buffer addresses for verification of reuse
    pub fn buffer_addresses(&self) -> (usize, usize) {
        (
            self.digests_buffer.contents() as usize,
            self.caps_buffer.contents() as usize,
        )
    }
}

// Note: PersistentMerkleBuffers is not Send/Sync by default due to Metal Buffer.
// In practice, Metal buffers can be safely shared across threads when using
// proper synchronization (command buffer dependencies).
unsafe impl Send for PersistentMerkleBuffers {}
unsafe impl Sync for PersistentMerkleBuffers {}
