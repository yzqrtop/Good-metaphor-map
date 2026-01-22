import json
from pathlib import Path
import argparse
import pandas as pd


def build_gold(scored_path: Path, output_path: Path):
    """Build gold dataset from scored results, take highest score record for each (E, V) pair"""
    # Read scored results
    df = pd.read_json(scored_path, lines=True)
    
    print(f"Read {len(df)} records")
    
    # Group by (E, V), take record with highest MEF score
    best = df.loc[df.groupby(["E", "V"])["MEF"].idxmax()]
    
    print(f"Building gold dataset, total {len(best)} records")
    
    # Write to output file
    best.to_json(output_path, orient="records", lines=True, force_ascii=False)
    
    print(f"Gold dataset construction completed, saved to {output_path}")


def merge_best():
    """Command line entry function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=Path, required=True, dest="scored_path", help="Scored input file path")
    parser.add_argument("--out", type=Path, required=True, dest="output_path", help="Output file path")
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build gold dataset
    build_gold(args.scored_path, args.output_path)


if __name__ == "__main__":
    merge_best()