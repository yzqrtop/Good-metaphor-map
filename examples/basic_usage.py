"""
GoodM Basic Usage Example

Demonstrates how to:
1. Create GoodMTuple
2. Use GenerationPipeline to generate seven-tuples
3. Use MEF for evaluation
4. Use IntentPredictor for intent prediction
"""

import asyncio
from goodm import GoodMTuple, GenerationPipeline, MEF, IntentPredictor
from goodm.models.llm_wrappers import QwenWrapper


async def main():
    print("=" * 60)
    print("GoodM Basic Usage Example")
    print("=" * 60)
    
    # Example 1: Create GoodMTuple
    print("\n[Example 1] Create GoodMTuple")
    print("-" * 40)
    
    tuple_obj = GoodMTuple(
        E="竹篮",
        V="一场空",
        EE="用竹子编织的篮子，通常用来盛东西",
        VE="什么都没有，完全落空",
        C="无法保持或留住",
        Sc="努力付出但没有任何收获的场景",
        Sy="徒劳无功，白费力气"
    )
    
    print(f"Created seven-tuple: {tuple_obj}")
    print(f"Validation result: {tuple_obj.validate()}")
    print(f"Prefix format:\n{tuple_obj.to_prefix()}")
    
    # Example 2: Use MEF for evaluation
    print("\n[Example 2] MEF Evaluation")
    print("-" * 40)
    
    mef = MEF()
    scores = mef.evaluate(tuple_obj)
    
    print(f"MEF total score: {scores['MEF']:.4f}")
    print("Individual metric scores:")
    for i in range(1, 8):
        print(f"  f{i}: {scores[f'f{i}']:.4f}")
    
    # Example 3: Intent prediction (without GoodM enhancement)
    print("\n[Example 3] Intent Prediction (Baseline)")
    print("-" * 40)
    
    # Note: Actual use requires loading pre-trained model
    # predictor = IntentPredictor(use_goodm_features=False)
    # result = predictor.predict("竹篮打水一场空")
    # print(f"Prediction result: {result}")
    
    print("Intent prediction example (requires pre-trained model):")
    print("  Input: '竹篮打水一场空'")
    print("  Expected output: warn (warning not to waste effort)")
    
    # Example 4: Use GenerationPipeline to generate
    print("\n[Example 4] Use GenerationPipeline to generate seven-tuple")
    print("-" * 40)
    
    print("Code example:")
    print("""
    # Initialize LLM client
    llm = QwenWrapper(model_name="qwen-plus")
    
    # Create generation pipeline
    pipeline = GenerationPipeline(
        llm_client=llm,
        temperature=0.7,
        consensus_rounds=3
    )
    
    # Generate seven-tuple
    result = await pipeline.generate(E="竹篮", V="一场空")
    
    if result:
        print(f"Generation successful: {result}")
    else:
        print("Generation failed, requires manual correction")
    """)
    
    print("\n" + "=" * 60)
    print("Example execution completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())