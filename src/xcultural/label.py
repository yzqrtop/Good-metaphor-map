import json
from pathlib import Path
import argparse

from src.xcultural.labeler import culture_label


def process_record(record):
    """Add cross-cultural label to single record"""
    try:
        ee = record.get("EE", "")
        ve = record.get("VE", "")
        
        # Add cultural label
        label = culture_label(ee, ve)
        record["culture_label"] = label
        
        return record
    except Exception as e:
        print(f"Error processing record: {e}")
        record["culture_label"] = "unknown"
        return record


def batch_label(input_path: Path, output_path: Path):
    """Batch add cross-cultural labels to records"""
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
    
    # Batch processing
    results = []
    for i, record in enumerate(records):
        result = process_record(record)
        results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"Processing progress: {i + 1}/{len(records)}")
    
    # Count label distribution
    label_counts = {"universal": 0, "culture-specific": 0, "adaptable": 0, "unknown": 0}
    for result in results:
        label = result.get("culture_label", "unknown")
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts["unknown"] += 1
    
    print(f"Label distribution: {label_counts}")
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Labeling completed, processed {len(results)} records saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=Path, required=True, dest="input_path", help="Input file path")
    parser.add_argument("--out", type=Path, required=True, dest="output_path", help="Output file path")
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 运行批量标签
    batch_label(args.input_path, args.output_path)