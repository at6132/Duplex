"""
CLI entry point for evaluating trained models.

Usage:
    # Compare baseline vs workspace
    python scripts/evaluate.py --baseline_ckpt checkpoints/baseline_phase2/final.pt \
                               --workspace_ckpt checkpoints/workspace_phase2/final.pt \
                               --test_data generated_data/test.jsonl

    # Evaluate single model
    python scripts/evaluate.py --checkpoint checkpoints/workspace_phase2/final.pt \
                               --test_data generated_data/test.jsonl

    # Run workspace size ablation
    python scripts/evaluate.py --ablation workspace_size \
                               --workspace_ckpt checkpoints/workspace_phase2/final.pt \
                               --test_data generated_data/test.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import CharTokenizer
from eval.benchmark import Benchmark


def load_test_samples(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def run_single_eval(args):
    tokenizer = CharTokenizer()
    bench = Benchmark(tokenizer, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    model, model_type = bench.load_model(args.checkpoint)
    test_samples = load_test_samples(args.test_data)

    if args.n_samples:
        test_samples = test_samples[:args.n_samples]

    print(f"\nEvaluating {model_type} model on {len(test_samples)} samples\n")

    if model_type == "baseline":
        metrics, raw = bench.evaluate_baseline(model, test_samples)
    else:
        metrics, raw = bench.evaluate_workspace(model, test_samples)

    print(f"\nResults:")
    print(f"  Revision Accuracy:  {metrics.revision_accuracy:.1%}")
    print(f"  Contradiction Rate: {metrics.contradiction_rate:.1%}")
    print(f"  Update Latency:     {metrics.mean_update_latency:.1f} chars")
    print(f"  N Samples:          {metrics.n_samples}")

    if metrics.per_task:
        print(f"\nPer-task breakdown:")
        for task, vals in sorted(metrics.per_task.items()):
            print(f"  {task}: acc={vals['revision_accuracy']:.1%}, "
                  f"contradiction={vals['contradiction_rate']:.1%}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{model_type}_results.json"), "w") as f:
        json.dump(raw, f, indent=2)


def run_comparison(args):
    tokenizer = CharTokenizer()
    bench = Benchmark(tokenizer, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    test_samples = load_test_samples(args.test_data)
    if args.n_samples:
        test_samples = test_samples[:args.n_samples]

    bench.compare(
        args.baseline_ckpt,
        args.workspace_ckpt,
        test_samples,
        output_dir=args.output_dir,
    )


def run_ablation_no_update(args):
    """Ablation: workspace model with update mechanism disabled."""
    tokenizer = CharTokenizer()
    bench = Benchmark(tokenizer, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    model, _ = bench.load_model(args.workspace_ckpt)
    test_samples = load_test_samples(args.test_data)
    if args.n_samples:
        test_samples = test_samples[:args.n_samples]

    print("\n=== Ablation: No Workspace Update (update_after_step=999999) ===")
    metrics_no_update, _ = bench.evaluate_workspace(
        model, test_samples, update_after_step=999999
    )

    print("\n=== Control: Normal Workspace Update ===")
    metrics_normal, _ = bench.evaluate_workspace(
        model, test_samples, update_after_step=5
    )

    print(f"\n{'Metric':<30} {'No Update':>12} {'With Update':>12}")
    print("-" * 60)
    print(f"{'Revision Accuracy':<30} {metrics_no_update.revision_accuracy:>11.1%} {metrics_normal.revision_accuracy:>11.1%}")
    print(f"{'Contradiction Rate':<30} {metrics_no_update.contradiction_rate:>11.1%} {metrics_normal.contradiction_rate:>11.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate duplex LM models")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single model checkpoint to evaluate")
    parser.add_argument("--baseline_ckpt", type=str, default=None,
                        help="Baseline checkpoint for comparison")
    parser.add_argument("--workspace_ckpt", type=str, default=None,
                        help="Workspace checkpoint for comparison")
    parser.add_argument("--test_data", type=str, default="generated_data/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit number of test samples")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_update"],
                        help="Run a specific ablation study")
    args = parser.parse_args()

    if args.ablation == "no_update":
        if not args.workspace_ckpt:
            print("Error: --workspace_ckpt required for no_update ablation")
            sys.exit(1)
        run_ablation_no_update(args)
    elif args.baseline_ckpt and args.workspace_ckpt:
        run_comparison(args)
    elif args.checkpoint:
        run_single_eval(args)
    else:
        print("Error: provide --checkpoint for single eval, or --baseline_ckpt + --workspace_ckpt for comparison")
        sys.exit(1)


if __name__ == "__main__":
    main()
