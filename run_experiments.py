#!/usr/bin/env python3
"""
GoodM experiment runner.

Usage:
    python run_experiments.py --rq 1 --ctrc data/ctrc.json
    python run_experiments.py --all --ctrc ... --intent ... --ood ...
    python run_experiments.py --rq 1 --ctrc ... --model gpt-4 --output results/rq1.json

RQ ↔ dataset mapping follows Table 6 of the paper:
    RQ1 CTRC test · RQ2 CTRC-Intent · RQ3 CTRC hard subset
    RQ4 CTRC test + hard subset · RQ5 CTRC dev · RQ6 OOD set
"""

import argparse
import json
from pathlib import Path

from embeddings.bertscore import BERTScorer
from embeddings.deepmets import DeepMetS
from embeddings.jina_embedder import JinaEmbedder
from experiments import (
    RQ1TASCloze,
    RQ2IntentPrediction,
    RQ3ConstraintVerification,
    RQ4HardFrequency,
    RQ5AblationStudy,
    RQ6ZeroShotOOD,
)
from llm_wrappers import create_model


def _build_model(args):
    return create_model(args.model) if args.model else None


def run_rq1(args, llm_client, embedder):
    return RQ1TASCloze(
        llm_client=llm_client, embedder=embedder,
        dataset_path=args.ctrc, limit=args.limit,
    ).run()


def run_rq2(args, llm_client, embedder):
    return RQ2IntentPrediction(
        llm_client=llm_client, embedder=embedder,
        dataset_path=args.intent, limit=args.limit,
    ).run()


def run_rq3(args, llm_client, embedder):
    return RQ3ConstraintVerification(
        llm_client=llm_client, embedder=embedder,
        dataset_path=args.ctrc, limit=args.limit,
    ).run()


def run_rq4(args, llm_client, embedder):
    return RQ4HardFrequency(
        llm_client=llm_client, embedder=embedder,
        dataset_path=args.ctrc, limit=args.limit,
    ).run()


def run_rq5(args, llm_client, embedder):
    return RQ5AblationStudy(
        llm_client=llm_client, embedder=embedder,
        dataset_path=args.ctrc, limit=args.limit,
    ).run()


def run_rq6(args, llm_client, embedder):
    return RQ6ZeroShotOOD(
        llm_client=llm_client, embedder=embedder,
        dataset_path=args.ood, limit=args.limit,
        bertscore=getattr(args, "bertscore_scorer", None),
        deepmets=getattr(args, "deepmets_model", None),
    ).run()


RUNNERS = {1: run_rq1, 2: run_rq2, 3: run_rq3, 4: run_rq4, 5: run_rq5, 6: run_rq6}


def main():
    parser = argparse.ArgumentParser(description="GoodM Experiment Runner")
    parser.add_argument("--rq", type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", type=str,
                        help="LLM name (gpt-4, qwen2.5-7b-instruct, ...)")
    parser.add_argument("--ctrc", type=str, help="CTRC dataset JSON (RQ1/3/4/5)")
    parser.add_argument("--intent", type=str, help="CTRC-Intent JSON (RQ2)")
    parser.add_argument("--ood", type=str, help="Golden OOD JSON (RQ6)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N instances")
    parser.add_argument("--output", type=str, help="Path to save the results JSON")
    parser.add_argument("--embedder", type=str, default="jina-embeddings-v3")
    parser.add_argument("--bertscore", type=str, default=None,
                        help="Enable BERTScore with this encoder (e.g. bert-base-chinese)")
    parser.add_argument("--deepmets", type=str, default=None,
                        help="DeepMet-S checkpoint directory (enables deepmets_sim)")
    args = parser.parse_args()

    if not args.rq and not args.all:
        parser.print_help()
        print("\nError: specify --rq [1-6] or --all")
        return

    llm_client = _build_model(args)
    embedder = JinaEmbedder(model_name=args.embedder)
    print(f"Embedding model: {embedder.model_name}")
    args.bertscore_scorer = BERTScorer(model_type=args.bertscore) if args.bertscore else None
    args.deepmets_model = DeepMetS(checkpoint=args.deepmets) if args.deepmets else None

    targets = [1, 2, 3, 4, 5, 6] if args.all else [args.rq]
    all_results = {}
    for rq in targets:
        print("=" * 60)
        print(f"RQ{rq}")
        print("=" * 60)
        results = RUNNERS[rq](args, llm_client, embedder)
        all_results[f"RQ{rq}"] = results
        print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
