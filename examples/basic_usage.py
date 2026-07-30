#!/usr/bin/env python3
"""
GoodM Basic Usage Example

Demonstrates basic usage of the GoodM framework:
1. Create GoodMTuple
2. Use processing operators
3. Verify constraints
4. Load datasets
"""

from core.schema import GoodMTuple
from operators import (
    EntityExpansion,
    VehicleExpansion,
    CommonalityExtraction,
    SceneGeneralization,
    Synthesis
)
from constraints import ConstraintVerifier
from embeddings import JinaEmbedder


def main():
    print("=" * 60)
    print("GoodM Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Create a GoodMTuple manually
    print("\n[Step 1] Create GoodMTuple")
    print("-" * 40)
    
    tuple_obj = GoodMTuple(
        E="bamboo basket",
        V="all in vain",
        EE="A basket woven from bamboo, with loose structure that cannot hold water",
        VE="Completely empty, achieving nothing despite effort",
        C="Unable to retain or preserve something valuable",
        Sc="A situation where significant effort produces no meaningful results",
        Sy="Futile effort is wasted labor; one should recognize when persistence is pointless"
    )
    
    print(f"Created tuple: {tuple_obj}")
    print(f"Valid: {tuple_obj.validate()}")
    print(f"\nPrefix format:\n{tuple_obj.to_prefix()}")
    
    # Step 2: Use processing operators
    print("\n[Step 2] Use Processing Operators")
    print("-" * 40)
    
    # Initialize embedder
    embedder = JinaEmbedder()
    
    # Expand entity
    entity_expander = EntityExpansion(embedder=embedder)
    EE = entity_expander.expand("bamboo basket")
    print(f"Entity Expansion (EE): {EE[:100]}...")
    
    # Expand vehicle
    vehicle_expander = VehicleExpansion(embedder=embedder)
    VE = vehicle_expander.expand("all in vain")
    print(f"Vehicle Expansion (VE): {VE[:100]}...")
    
    # Extract commonality
    commonality_extractor = CommonalityExtraction(embedder=embedder)
    C = commonality_extractor.extract(EE, VE)
    print(f"Commonality (C): {C[:100]}...")
    
    # Generate scene
    scene_generator = SceneGeneralization(embedder=embedder)
    Sc = scene_generator.generalize(EE, VE)
    print(f"Scene (Sc): {Sc[:100]}...")
    
    # Synthesize
    synthesizer = Synthesis(embedder=embedder)
    Sy = synthesizer.synthesize(Sc, C)
    print(f"Synthetics (Sy): {Sy[:100]}...")
    
    # Create generated tuple
    generated_tuple = GoodMTuple(
        E="bamboo basket",
        V="all in vain",
        EE=EE,
        VE=VE,
        C=C,
        Sc=Sc,
        Sy=Sy
    )
    
    # Step 3: Verify constraints
    print("\n[Step 3] Verify Constraints")
    print("-" * 40)
    
    verifier = ConstraintVerifier(embedder=embedder)
    results = verifier.verify_all(generated_tuple)
    
    print(f"All constraints passed: {results['overall']['passed']}")
    print(f"CPR (Constraint Pass Rate): {results['overall']['pass_rate']:.2%}")
    print(f"\nIndividual constraint results:")
    for constraint, result in results.items():
        if isinstance(result, dict) and 'passed' in result:
            status = "✓" if result['passed'] else "✗"
            print(f"  {constraint}: {status} (score: {result.get('sim_EE_E', result.get('sim_C_EE', 'N/A')):.3f})")
    
    # Step 4: Compute CPR for multiple tuples
    print("\n[Step 4] Compute CPR for Multiple Tuples")
    print("-" * 40)
    
    test_tuples = [
        generated_tuple,
        tuple_obj,  # Original manually created tuple
    ]
    
    cpr = verifier.compute_cpr(test_tuples)
    print(f"CPR for {len(test_tuples)} tuples: {cpr:.2%}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()