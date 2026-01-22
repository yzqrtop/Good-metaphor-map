import json
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor

from src.minieval.evaluator import MiniEval


def process_record(record, evaluator):
    """Process single record"""
    return evaluator.score_one(record)


def batch_score(input_path: Path, output_path: Path):
    """Batch score records"""
    # Read input file
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError:
                continue
    
    print(f"Read {len(records)} records")
    
    # Initialize evaluator
    evaluator = MiniEval()
    
    # Batch processing
    results = []
    batch_size = 100
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        batch_results = [process_record(record, evaluator) for record in batch]
        results.extend(batch_results)
        
        print(f"Processing progress: {min(i+batch_size, len(records))}/{len(records)}")
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Scoring completed, processed {len(results)} records saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=Path, required=True, dest="input_path", help="Input file path")
    parser.add_argument("--out", type=Path, required=True, dest="output_path", help="Output file path")
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 运行批量打分
    batch_score(args.input_path, args.output_path)