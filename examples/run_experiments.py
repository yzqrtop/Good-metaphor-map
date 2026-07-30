#!/usr/bin/env python3
"""
GoodM Experiment Runner Example

Demonstrates how to run RQ experiments using the GoodM framework.
"""

import argparse
from experiments import (
    RQ1TASCloze,
    RQ2IntentPrediction,
    RQ3ConstraintVerification,
    RQ4HardFrequency,
    RQ5AblationStudy,
    RQ6ZeroShotOOD
)


def run_rq1():
    """Run RQ1: TAS Cloze Evaluation"""
    print("Running RQ1: TAS Cloze Evaluation...")
    experiment = RQ1TASCloze()
    results = experiment.run()
    print(f"Accuracy: {results['results']['accuracy']:.4f}")
    print(f"Macro-F1: {results['results']['macro_f1']:.4f}")
    print(f"BERTScore: {results['results']['bert_score']:.4f}")
    print(f"CPR: {results['results']['cpr']:.4f}")


def run_rq5():
    """Run RQ5: Ablation Study"""
    print("Running RQ5: Ablation Study...")
    experiment = RQ5AblationStudy()
    results = experiment.run()
    
    print("\nBaseline Results:")
    print(f"  Accuracy: {results['results']['baseline']['accuracy']:.4f}")
    print(f"  CPR: {results['results']['baseline']['cpr']:.4f}")
    
    print("\nAblation Results:")
    for component, ablation_result in results['results']['ablations'].items():
        print(f"  {component}: Accuracy={ablation_result['accuracy']:.4f}, CPR={ablation_result['cpr']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="GoodM Experiment Example")
    parser.add_argument('--rq', type=int, choices=[1, 2, 3, 4, 5, 6], default=1,
                        help='Which RQ experiment to run')
    args = parser.parse_args()
    
    rq_map = {
        1: run_rq1,
        5: run_rq5,
    }
    
    if args.rq in rq_map:
        rq_map[args.rq]()
    else:
        print(f"RQ{args.rq} requires a dataset file. Use run_experiments.py with --dataset parameter.")
        print("Example: python run_experiments.py --rq 2 --dataset data/ctrc_intent.json")


if __name__ == "__main__":
    main()