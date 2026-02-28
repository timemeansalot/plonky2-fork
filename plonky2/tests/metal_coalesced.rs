//! Correctness tests for coalesced Merkle tree GPU implementation.
//!
//! Tree heights >= 21 route to the coalesced shader path.
//! Tree height 20 routes to the linear+threadgroup path.
//! Both paths are tested here to ensure the GPU produces structurally
//! correct Merkle trees and that digests are non-zero.

#[cfg(feature = "metal")]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::hash::merkle_tree::MerkleTree;
    use plonky2::hash::poseidon::PoseidonHash;

    /// Goldilocks field order: 2^64 - 2^32 + 1
    const GL_ORDER: u64 = 0xFFFFFFFF00000001;

    /// Build a Merkle tree at the given height and cap_height, then verify:
    /// 1. Cap length matches 2^cap_height
    /// 2. Digest count matches the expected formula
    /// 3. No digest slot is left as the zero hash (all were actually written)
    fn test_merkle_at_height(tree_height: usize, cap_height: usize) {
        let leaf_count = 1usize << tree_height;
        let leaf_size = 8;
        let total_elements = leaf_count * leaf_size;

        // Deterministic non-trivial leaf data.
        let leaves: Vec<GoldilocksField> = (0..total_elements)
            .map(|i| {
                GoldilocksField::from_canonical_u64(
                    (i as u64 * 7 + 13) % GL_ORDER,
                )
            })
            .collect();

        println!(
            "Building Merkle tree: tree_height={}, cap_height={}, leaf_count={}, leaf_size={}",
            tree_height, cap_height, leaf_count, leaf_size,
        );

        let tree = MerkleTree::<GoldilocksField, PoseidonHash>::new_from_1d(
            leaves,
            leaf_size,
            cap_height,
        );

        // Check cap length.
        assert_eq!(
            tree.cap.0.len(),
            1 << cap_height,
            "Cap length mismatch at tree_height={}, cap_height={}",
            tree_height,
            cap_height,
        );

        // Check digest count.
        // In this fork: num_digests = 2 * (leaves_len - (1 << cap_height))
        // Which equals (2 * (subtree_leaves - 1)) * num_subtrees.
        let subtree_leaves = leaf_count >> cap_height;
        let expected_digests = 2 * (subtree_leaves - 1) * (1 << cap_height);
        assert_eq!(
            tree.digests.len(),
            expected_digests,
            "Digest count mismatch at tree_height={}, cap_height={}",
            tree_height,
            cap_height,
        );

        // Verify no digest is the zero hash (GPU actually wrote every slot).
        let zero = <PoseidonHash as plonky2::plonk::config::Hasher<GoldilocksField>>::Hash::default();
        let zero_count = tree.digests.iter().filter(|d| **d == zero).count();
        assert_eq!(
            zero_count, 0,
            "Found {} zero digests out of {} at tree_height={}, cap_height={}",
            zero_count,
            tree.digests.len(),
            tree_height,
            cap_height,
        );

        // Verify no cap entry is the zero hash.
        let zero_cap_count = tree.cap.0.iter().filter(|d| **d == zero).count();
        assert_eq!(
            zero_cap_count, 0,
            "Found {} zero cap entries at tree_height={}, cap_height={}",
            zero_cap_count,
            tree_height,
            cap_height,
        );

        println!(
            "  OK: {} digests, {} cap entries, no zeros",
            tree.digests.len(),
            tree.cap.0.len(),
        );
    }

    /// Coalesced path: tree_height=21, cap_height=4
    #[test]
    fn test_coalesced_height_21_cap_4() {
        test_merkle_at_height(21, 4);
    }

    /// Coalesced path: tree_height=22, cap_height=4 (memory-intensive, ~256MB leaves)
    #[test]
    fn test_coalesced_height_22_cap_4() {
        test_merkle_at_height(22, 4);
    }

    /// Coalesced path: tree_height=21, cap_height=1
    #[test]
    fn test_coalesced_height_21_cap_1() {
        test_merkle_at_height(21, 1);
    }

    /// Linear+threadgroup path: tree_height=20, cap_height=4
    #[test]
    fn test_linear_height_20_cap_4() {
        test_merkle_at_height(20, 4);
    }
}
