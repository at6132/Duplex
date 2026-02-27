"""
Generate synthetic train/val/test datasets and save as JSONL files.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --output_dir generated_data --n_train 50000 --n_val 5000 --n_test 5000
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tasks import generate_dataset, TASK_GENERATORS


def save_jsonl(samples: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic duplex LM data")
    parser.add_argument("--output_dir", type=str, default="generated_data")
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_types", type=str, nargs="+", default=None,
                        help=f"Task types to include. Available: {list(TASK_GENERATORS.keys())}")
    args = parser.parse_args()

    task_types = args.task_types or list(TASK_GENERATORS.keys())
    print(f"Generating data with task types: {task_types}")

    print("\nGenerating train set...")
    train = generate_dataset(args.n_train, task_types=task_types, seed=args.seed)
    save_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))

    print("Generating val set...")
    val = generate_dataset(args.n_val, task_types=task_types, seed=args.seed + 1)
    save_jsonl(val, os.path.join(args.output_dir, "val.jsonl"))

    print("Generating test set...")
    test = generate_dataset(args.n_test, task_types=task_types, seed=args.seed + 2)
    save_jsonl(test, os.path.join(args.output_dir, "test.jsonl"))

    # Print stats
    from collections import Counter
    type_counts = Counter(s["task_type"] for s in train)
    print("\nTrain set task distribution:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    print(f"\nTotal: {args.n_train + args.n_val + args.n_test} samples")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
