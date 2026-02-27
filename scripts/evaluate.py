"""
Evaluation script for Duplex-1-1.7B vs vanilla Qwen3-1.7B.

Usage:
    python scripts/evaluate.py \
        --duplex_ckpt checkpoints/duplex-1-1.7b/final.pt \
        --n_samples 200
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from duplex.config import DuplexConfig
from duplex.duplex_model import DuplexModel
from duplex.inference.generate import load_duplex_model


def evaluate_sample(continuation: str, expected: dict) -> dict:
    text_lower = continuation.lower()
    correct = str(expected.get("correct_value", "")).lower()
    wrong = str(expected.get("wrong_value", "")).lower()
    extra = str(expected.get("extra_correct", "")).lower()

    has_correct = correct in text_lower if correct else False
    has_wrong = wrong in text_lower if wrong else False
    has_extra = extra in text_lower if extra else True  # default True if no extra

    revision_accurate = has_correct and has_extra
    has_contradiction = has_wrong

    latency = -1
    if has_correct:
        idx = text_lower.find(correct)
        if idx >= 0:
            latency = idx

    return {
        "revision_accurate": revision_accurate,
        "has_contradiction": has_contradiction,
        "update_latency": latency,
    }


def run_duplex_eval(model: DuplexModel, test_samples: list[dict], n_samples: int) -> dict:
    results = []
    per_task = defaultdict(list)

    for sample in tqdm(test_samples[:n_samples], desc="Evaluating Duplex"):
        full_text, _ = model.generate_with_update(
            prompt_text=sample["prompt"],
            max_new_tokens=150,
            temperature=0.7,
            correction_text=sample["correction"],
            correction_after_tokens=10,
        )

        expected = sample.get("expected_values", {})
        m = evaluate_sample(full_text, expected)
        m["task_type"] = sample["task_type"]
        results.append(m)
        per_task[sample["task_type"]].append(m)

    n = len(results)
    rev_acc = sum(1 for r in results if r["revision_accurate"]) / max(1, n)
    contra = sum(1 for r in results if r["has_contradiction"]) / max(1, n)
    lats = [r["update_latency"] for r in results if r["update_latency"] >= 0]
    mean_lat = sum(lats) / max(1, len(lats))

    task_summary = {}
    for task, task_results in per_task.items():
        tn = len(task_results)
        task_summary[task] = {
            "revision_accuracy": sum(1 for r in task_results if r["revision_accurate"]) / tn,
            "contradiction_rate": sum(1 for r in task_results if r["has_contradiction"]) / tn,
            "n": tn,
        }

    return {
        "revision_accuracy": rev_acc,
        "contradiction_rate": contra,
        "mean_update_latency": mean_lat,
        "n_samples": n,
        "per_task": task_summary,
    }


def run_ablation(model: DuplexModel, test_samples: list[dict], n_samples: int) -> dict:
    """Run with update disabled (correction_after_tokens=999999)."""
    results = []

    for sample in tqdm(test_samples[:n_samples], desc="Ablation (no update)"):
        full_text, _ = model.generate_with_update(
            prompt_text=sample["prompt"],
            max_new_tokens=150,
            temperature=0.7,
            correction_text=sample["correction"],
            correction_after_tokens=999999,
        )

        expected = sample.get("expected_values", {})
        m = evaluate_sample(full_text, expected)
        results.append(m)

    n = len(results)
    return {
        "revision_accuracy": sum(1 for r in results if r["revision_accurate"]) / max(1, n),
        "contradiction_rate": sum(1 for r in results if r["has_contradiction"]) / max(1, n),
        "n_samples": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duplex_ckpt", required=True)
    parser.add_argument("--qwen_path", default="models/qwen3-1.7b-base")
    parser.add_argument("--test_data", default="generated_data_duplex/test.jsonl")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    samples = []
    with open(args.test_data, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print("Loading Duplex-1-1.7B...")
    model = load_duplex_model(args.qwen_path, args.duplex_ckpt)

    print(f"\nEvaluating on {min(args.n_samples, len(samples))} samples\n")
    metrics = run_duplex_eval(model, samples, args.n_samples)

    print(f"\n{'='*50}")
    print(f"  Duplex-1-1.7B Results")
    print(f"{'='*50}")
    print(f"  Revision Accuracy:  {metrics['revision_accuracy']:.1%}")
    print(f"  Contradiction Rate: {metrics['contradiction_rate']:.1%}")
    print(f"  Update Latency:     {metrics['mean_update_latency']:.1f} chars")
    print(f"  N Samples:          {metrics['n_samples']}")
    if metrics.get("per_task"):
        print(f"\n  Per-task:")
        for task, vals in sorted(metrics["per_task"].items()):
            print(f"    {task}: acc={vals['revision_accuracy']:.1%}, "
                  f"contradiction={vals['contradiction_rate']:.1%}")
    print(f"{'='*50}")

    if args.ablation:
        print("\nRunning ablation (no workspace update)...")
        abl = run_ablation(model, samples, args.n_samples)
        print(f"\n{'Metric':<30} {'No Update':>12} {'With Update':>12}")
        print("-" * 55)
        print(f"{'Revision Accuracy':<30} {abl['revision_accuracy']:>11.1%} {metrics['revision_accuracy']:>11.1%}")
        print(f"{'Contradiction Rate':<30} {abl['contradiction_rate']:>11.1%} {metrics['contradiction_rate']:>11.1%}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "duplex_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
