#!/usr/bin/env python3
"""
GoodM Basic Usage Example

Demonstrates basic usage of the GoodM framework with real Chinese xielingyu (allegorical saying):
1. Create GoodMTuple with real data
2. Use processing operators
3. Verify constraints
4. Compute CPR
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
    
    # Step 1: Create a GoodMTuple with real xielingyu data
    print("\n[Step 1] Create GoodMTuple with real data")
    print("-" * 40)
    
    # Real xielingyu: 王婆卖瓜 - 自买自夸
    tuple_obj = GoodMTuple(
        E="王婆卖瓜",
        V="自买自夸",
        EE="王婆（古代对年长妇人的尊称）在集市上卖瓜，此处引申为自己夸赞自己的东西好，比喻自我吹嘘、自我标榜",
        VE="自己买自己的瓜，自己夸奖自己的货物，比喻自我炫耀、自我吹捧",
        C="对自己的东西或行为进行过度的夸奖和标榜，带有自吹自擂的意味",
        Sc="集市或市场交易场景中，卖方为了促销而过度吹嘘自己的商品质量或价值",
        Sy="讽刺那些自我吹嘘、夸大其词的行为，提醒人们要客观看待事物，不要盲目自夸"
    )
    
    print(f"Created tuple: {tuple_obj}")
    print(f"Valid: {tuple_obj.validate()}")
    print(f"\nPrefix format:\n{tuple_obj.to_prefix()}")
    
    # Step 2: Use processing operators (demonstrates the pipeline)
    print("\n[Step 2] Use Processing Operators")
    print("-" * 40)
    
    # Initialize embedder
    embedder = JinaEmbedder()
    
    # Expand entity (E → EE)
    entity_expander = EntityExpansion(embedder=embedder)
    EE_generated = entity_expander.expand("王婆卖瓜")
    print(f"Entity Expansion (EE): {EE_generated[:100]}...")
    
    # Expand vehicle (V → VE)
    vehicle_expander = VehicleExpansion(embedder=embedder)
    VE_generated = vehicle_expander.expand("自买自夸")
    print(f"Vehicle Expansion (VE): {VE_generated[:100]}...")
    
    # Extract commonality (EE × VE → C)
    commonality_extractor = CommonalityExtraction(embedder=embedder)
    C_generated = commonality_extractor.extract(EE_generated, VE_generated)
    print(f"Commonality (C): {C_generated[:100]}...")
    
    # Generate scene (EE × VE → Sc)
    scene_generator = SceneGeneralization(embedder=embedder)
    Sc_generated = scene_generator.generalize(EE_generated, VE_generated)
    print(f"Scene (Sc): {Sc_generated[:100]}...")
    
    # Synthesize (Sc × C → Sy)
    synthesizer = Synthesis(embedder=embedder)
    Sy_generated = synthesizer.synthesize(Sc_generated, C_generated)
    print(f"Synthetics (Sy): {Sy_generated[:100]}...")
    
    # Create generated tuple from operators
    generated_tuple = GoodMTuple(
        E="王婆卖瓜",
        V="自买自夸",
        EE=EE_generated,
        VE=VE_generated,
        C=C_generated,
        Sc=Sc_generated,
        Sy=Sy_generated
    )
    
    # Step 3: Verify constraints
    print("\n[Step 3] Verify Constraints")
    print("-" * 40)
    
    verifier = ConstraintVerifier(embedder=embedder)
    
    # Verify manually created tuple
    print("\nVerifying manually created tuple:")
    results = verifier.verify_all(tuple_obj)
    print(f"All constraints passed: {results['overall']['passed']}")
    print(f"CPR (Constraint Pass Rate): {results['overall']['pass_rate']:.2%}")
    
    print("\nVerifying generated tuple:")
    gen_results = verifier.verify_all(generated_tuple)
    print(f"All constraints passed: {gen_results['overall']['passed']}")
    print(f"CPR (Constraint Pass Rate): {gen_results['overall']['pass_rate']:.2%}")
    
    print("\nIndividual constraint results (manually created):")
    for constraint, result in results.items():
        if isinstance(result, dict) and 'passed' in result:
            status = "✓" if result['passed'] else "✗"
            faith_score = result.get('sim_EE_E', result.get('sim_C_EE', result.get('sim_Sy_C', 'N/A')))
            print(f"  {constraint}: {status} (score: {faith_score:.3f})")
    
    # Step 4: Compute CPR for multiple tuples
    print("\n[Step 4] Compute CPR for Multiple Tuples")
    print("-" * 40)
    
    # Use multiple real xielingyu examples
    test_tuples = [
        tuple_obj,  # 王婆卖瓜 - 自买自夸
        generated_tuple,  # Generated from operators
        GoodMTuple(
            E="井底之蛙",
            V="没见过大世面",
            EE="井底的青蛙，比喻见识短浅的人",
            VE="没有见过大的世面，形容见识狭隘",
            C="见识短浅、视野狭隘",
            Sc="局限于狭小空间或环境中",
            Sy="告诫人们不要固步自封，要开阔眼界"
        ),
        GoodMTuple(
            E="竹篮打水",
            V="一场空",
            EE="用竹篮打水，水会从缝隙中漏走",
            VE="做了事情但毫无结果",
            C="徒劳无功、白费力气",
            Sc="做无用功的场景",
            Sy="提醒人们选择正确的方法和方向"
        ),
    ]
    
    cpr = verifier.compute_cpr(test_tuples)
    print(f"CPR for {len(test_tuples)} real xielingyu tuples: {cpr:.2%}")
    
    # Show all E-V pairs
    print("\nDemonstrated xielingyu examples:")
    for i, t in enumerate(test_tuples, 1):
        print(f"  {i}. {t.E} — {t.V}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()