import json
from pathlib import Path
import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor

from src.generate.generator import GoodMGenerator


async def process_item(item, generator):
    """Process single item"""
    try:
        # Assume each item in input file has "idiom" field
        # In actual projects, may need to adjust based on input file format
        e = item.get("idiom", "")
        v = "Related metaphor vehicle"  # In actual projects, may need to get from input or generate
        
        if not e:
            return None
        
        result = await generator.build_one(e, v)
        return result
    except Exception as e:
        print(f"Error processing item: {e}")
        return None


async def batch_process(input_path: Path, output_path: Path, model_name: str, workers: int):
    """Batch process input file"""
    # Read input file
    items = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                items.append(item)
            except json.JSONDecodeError:
                continue
    
    print(f"Read {len(items)} items")
    
    # Initialize generator
    generator = GoodMGenerator(model_name)
    
    # Batch processing
    results = []
    batch_size = workers * 2
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        tasks = [process_item(item, generator) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend([r for r in batch_results if r is not None])
        
        print(f"Processing progress: {min(i+batch_size, len(items))}/{len(items)}")
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Processing completed, generated {len(results)} seven-tuples saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=Path, required=True, dest="input_path", help="Input file path")
    parser.add_argument("--out", type=Path, required=True, dest="output_path", help="Output file path")
    parser.add_argument("--model", type=str, default="qwen-plus", help="Model name")
    parser.add_argument("--workers", type=int, default=32, help="Number of workers")
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 运行批量处理
    asyncio.run(batch_process(args.input_path, args.output_path, args.model, args.workers))