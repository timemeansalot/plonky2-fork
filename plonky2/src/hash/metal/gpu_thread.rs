//! Dedicated GPU dispatch thread for Metal command buffer operations.
//!
//! Moves all `CommandQueue` usage to a single `std::thread`, fixing the thread
//! safety violation where multiple Rayon workers could concurrently submit
//! command buffers through the shared `CommandQueue`.
//!
//! Rayon workers send `GpuJob` messages via `mpsc::channel` and block on a
//! oneshot reply channel until the GPU thread completes the work.

use metal::objc::rc::autoreleasepool;
use metal::Buffer;
use once_cell::sync::Lazy;
use std::sync::mpsc;
use std::thread;

use crate::hash::hash_types::HashOut;
use plonky2_field::goldilocks_field::GoldilocksField;

type MerkleResult = (Vec<HashOut<GoldilocksField>>, Vec<HashOut<GoldilocksField>>);

/// Wrapper asserting Metal `Buffer` is `Send` for `StorageModeShared` on UMA.
///
/// # Safety
/// On Apple Silicon (UMA), `StorageModeShared` buffers are backed by a single
/// physical allocation accessible from both CPU and GPU. The pointer is stable
/// and does not move when sent across threads. All buffers created by
/// `MetalRuntime` use `StorageModeShared`, so this is safe for our usage.
pub(crate) struct SendableBuffer(pub Buffer);
unsafe impl Send for SendableBuffer {}

pub(crate) enum GpuJob {
    MerkleLinearThreadgroup {
        leaves_buffer: SendableBuffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
        reply: mpsc::Sender<MerkleResult>,
    },
    MerkleCoalesced {
        leaves_buffer: SendableBuffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
        reply: mpsc::Sender<MerkleResult>,
    },
}

pub(crate) struct GpuDispatcher {
    sender: mpsc::Sender<GpuJob>,
}

pub(crate) static GPU_DISPATCHER: Lazy<GpuDispatcher> = Lazy::new(GpuDispatcher::new);

impl GpuDispatcher {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel::<GpuJob>();

        thread::Builder::new()
            .name("metal-gpu-dispatch".into())
            .spawn(move || {
                autoreleasepool(|| {
                    Self::event_loop(rx);
                });
            })
            .expect("failed to spawn Metal GPU dispatch thread");

        GpuDispatcher { sender: tx }
    }

    fn event_loop(rx: mpsc::Receiver<GpuJob>) {
        use super::runtime::RUNTIME;

        while let Ok(job) = rx.recv() {
            autoreleasepool(|| match job {
                GpuJob::MerkleLinearThreadgroup {
                    leaves_buffer,
                    tree_height,
                    leaf_length,
                    cap_height,
                    reply,
                } => {
                    let result = RUNTIME.hash_merkle_tree_linear_threadgroup_buf_ho(
                        leaves_buffer.0,
                        tree_height,
                        leaf_length,
                        cap_height,
                    );
                    let _ = reply.send(result);
                }
                GpuJob::MerkleCoalesced {
                    leaves_buffer,
                    tree_height,
                    leaf_length,
                    cap_height,
                    reply,
                } => {
                    let result = RUNTIME.hash_merkle_tree_coalesced_buf_ho(
                        leaves_buffer.0,
                        tree_height,
                        leaf_length,
                        cap_height,
                    );
                    let _ = reply.send(result);
                }
            });
        }
        // Sender disconnected — GpuDispatcher was dropped. Thread exits cleanly.
    }

    /// Dispatch a linear+threadgroup Merkle tree hash job to the GPU thread.
    ///
    /// Blocks the caller until the GPU thread completes the work and returns
    /// the result via a oneshot reply channel.
    pub(crate) fn dispatch_merkle_linear_threadgroup(
        &self,
        leaves_buffer: Buffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
    ) -> MerkleResult {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(GpuJob::MerkleLinearThreadgroup {
                leaves_buffer: SendableBuffer(leaves_buffer),
                tree_height,
                leaf_length,
                cap_height,
                reply: tx,
            })
            .expect("GPU dispatch thread terminated unexpectedly");
        rx.recv()
            .expect("GPU dispatch thread dropped reply channel")
    }

    /// Dispatch a coalesced Merkle tree hash job to the GPU thread.
    ///
    /// Same blocking pattern as `dispatch_merkle_linear_threadgroup`.
    #[allow(dead_code)]
    pub(crate) fn dispatch_merkle_coalesced(
        &self,
        leaves_buffer: Buffer,
        tree_height: usize,
        leaf_length: usize,
        cap_height: usize,
    ) -> MerkleResult {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(GpuJob::MerkleCoalesced {
                leaves_buffer: SendableBuffer(leaves_buffer),
                tree_height,
                leaf_length,
                cap_height,
                reply: tx,
            })
            .expect("GPU dispatch thread terminated unexpectedly");
        rx.recv()
            .expect("GPU dispatch thread dropped reply channel")
    }
}
