//! End-to-end test: generate a proof using Metal GPU acceleration, verify on CPU.
//!
//! With `standard_recursion_config` (rate_bits=3, cap_height=4):
//!   - 8000 Poseidon hashes produce degree 2^13
//!   - LDE domain = 2^(13+3) = 2^16, so Metal NTT is exercised (threshold: log_n + rate_bits >= 16)
//!   - Merkle tree height = 16, so linear+threadgroup GPU Merkle is exercised (threshold: 13..=20)

#[cfg(feature = "metal")]
mod metal_proof_tests {
    use std::time::Instant;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::iop::witness::{PartialWitness, WitnessWrite};
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::PoseidonGoldilocksConfig;

    type F = GoldilocksField;
    type C = PoseidonGoldilocksConfig;
    const D: usize = 2;

    /// Build a circuit large enough to exercise the Metal NTT + Merkle paths.
    /// Uses many Poseidon hashes to inflate the circuit.
    #[test]
    fn test_metal_proof_roundtrip() {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // Build a chain of Poseidon hashes to create a large circuit.
        // Each Poseidon hash adds ~1 PoseidonGate (135 wires).
        // 8000 hashes yield degree 2^13, which is the minimum to exercise
        // both Metal NTT (log_n + rate_bits = 13 + 3 = 16 >= 16)
        // and Metal Merkle (tree_height = 16, in 13..=20 range).
        let initial = builder.add_virtual_target();
        let mut current = initial;
        for _ in 0..8000 {
            let hash_inputs = vec![current; 4];
            let hash_out = builder.hash_n_to_hash_no_pad::<
                plonky2::hash::poseidon::PoseidonHash,
            >(hash_inputs);
            current = hash_out.elements[0];
        }
        builder.register_public_input(current);

        let t0 = Instant::now();
        let data = builder.build::<C>();
        let build_time = t0.elapsed();
        let degree_bits = data.common.degree_bits();
        println!(
            "Circuit built: degree 2^{}, build time: {:.2?}",
            degree_bits, build_time,
        );

        let mut pw = PartialWitness::new();
        pw.set_target(initial, F::from_canonical_u64(42));

        let t1 = Instant::now();
        let proof = data.prove(pw).expect("Proof generation failed");
        let prove_time = t1.elapsed();
        println!("Proof generated in {:.2?}", prove_time);

        let t2 = Instant::now();
        data.verify(proof).expect("Proof verification failed");
        let verify_time = t2.elapsed();
        println!("Proof verified in {:.2?}", verify_time);

        println!(
            "End-to-end Metal GPU proof: degree 2^{}, build {:.2?}, prove {:.2?}, verify {:.2?}",
            degree_bits, build_time, prove_time, verify_time,
        );
    }
}
