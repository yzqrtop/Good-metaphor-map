#!/usr/bin/env python3
"""
GoodM Experiment Runner

Main entry point for running all RQ experiments.
Usage:
    python run_experiments.py --rq [1-6] --model [model_name]
    python run_experiments.py --all --model gpt-4
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from experiments import (
    RQ1TASCloze,
    RQ2IntentPrediction,
    RQ3ConstraintVerification,
    RQ4HardFrequency,
    RQ5AblationStudy,
    RQ6ZeroShotOOD
)
from embeddings import JinaEmbedder
from models.llm_wrappers import create_model


def run_rq1(args):
    """Run RQ1: TAS Cloze Evaluation"""
    print("=" * 60)
    print("RQ1: TAS Cloze Evaluation")
    print("=" * 60)
    
    embedder = JinaEmbedder()
    llm_client = create_model(args.model) if args.model else None
    
    experiment = RQ1TASCloze(
        llm_client=llm_client,
        embedder=embedder,
        dataset_path=args.dataset
    )
    
    results = experiment.run()
    print_results(results)
    return results


def run_rq2(args):
    """Run RQ2: Intent Prediction"""
    print("=" * 60)
    print("RQ2: Intent Prediction")
    print("=" * 60)
    
    embedder = JinaEmbedder()
    llm_client = create_model(args.model) if args.model else None
    
    experiment = RQ2IntentPrediction(
        llm_client=llm_client,
        embedder=embedder,
        dataset_path=args.dataset
    )
    
    results = experiment.run()
    print_results(results)
    return results


def run_rq3(args):
    """Run RQ3: Constraint Verification"""
    print("=" * 60)
    print("RQ3: Constraint Verification")
    print("=" * 60)
    
    embedder = JinaEmbedder()
    llm_client = create_model(args.model) if args.model else None
    
    experiment = RQ3ConstraintVerification(
        llm_client=llm_client,
        embedder=embedder,
        dataset_path=args.dataset
    )
    
    results = experiment.run()
    print_results(results)
    return results


def run_rq4(args):
    """Run RQ4: Hard & Frequency Analysis"""
    print("=" * 60)
    print("RQ4: Hard & Frequency Analysis")
    print("=" * 60)
    
    embedder = JinaEmbedder()
    llm_client = create_model(args.model) if args.model else None
    
    experiment = RQ4HardFrequency(
        llm_client=llm_client,
        embedder=embedder,
        dataset_path=args.dataset
    )
    
    results = experiment.run()
    print_results(results)
    return results


def run_rq5(args):
    """Run RQ5: Ablation Study"""
    print("=" * 60)
    print("RQ5: Ablation Study")
    print("=" * 60)
    
    embedder = JinaEmbedder()
    llm_client = create_model(args.model) if args.model else None
    
    experiment = RQ5AblationStudy(
        llm_client=llm_client,
        embedder=embedder,
        dataset_path=args.dataset
    )
    
    results = experiment.run()
    print_results(results)
    return results


def run_rq6(args):
    """Run RQ6: Zero-shot OOD Evaluation"""
    print("=" * 60)
    print("RQ6: Zero-shot OOD Evaluation")
    print("=" * 60)
    
    embedder = JinaEmbedder()
    llm_client = create_model(args.model) if args.model else None
    
    experiment = RQ6ZeroShotOOD(
        llm_client=llm_client,
        embedder=embedder,
        dataset_path=args.dataset
    )
    
    results = experiment.run()
    print_results(results)
    return results


def print_results(results: dict):
    """Print experiment results"""
    print("\nResults:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


def save_results(results: dict, output_path: str):
    """Save results to file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GoodM Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --rq 1 --model gpt-4
  python run_experiments.py --all --model qwen-plus
  python run_experiments.py --rq 5 --dataset data/ctrc.json --output results/rq5.json
        """
    )
    
    parser.add_argument(
        '--rq',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Run specific RQ experiment (1-6)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='LLM model name (e.g., gpt-4, qwen-plus, claude-3.5-sonnet)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save results'
    )
    
    parser.add_argument(
        '--embedder',
        type=str,
        default='jina-embeddings-v3',
        help='Embedder model name'
    )
    
    args = parser.parse_args()
    
    if not args.rq and not args.all:
        parser.print_help()
        print("\nError: Please specify --rq [1-6] or --all")
        return
    
    # Run experiments
    all_results = {}
    
    if args.all or args.rq == 1:
        all_results['RQ1'] = run_rq1(args)
    
    if args.all or args.rq == 2:
        all_results['RQ2'] = run_rq2(args)
    
    if args.all or args.rq == 3:
        all_results['RQ3'] = run_rq3(args)
    
    if args.all or args.rq == 4:
        all_results['RQ4'] = run_rq4(args)
    
    if args.all or args.rq == 5:
        all_results['RQ5'] = run_rq5(args)
    
    if args.all or args.rq == 6:
        all_results['RQ6'] = run_rq6(args)
    
    # Save results if output path specified
    if args.output:
        save_results(all_results, args.output)


if __name__ == "__main__":
    main()