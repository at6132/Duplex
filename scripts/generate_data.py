"""Generate synthetic training data for Duplex-1."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duplex.data.tasks import generate_dataset, TASK_GENERATORS


def save_jsonl(samples: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="generated_data_duplex")
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    types = list(TASK_GENERATORS.keys())
    print(f"Task types: {types}")

    print("\nGenerating train set...")
    train = generate_dataset(args.n_train, seed=args.seed)
    save_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))

    print("Generating val set...")
    val = generate_dataset(args.n_val, seed=args.seed + 1)
    save_jsonl(val, os.path.join(args.output_dir, "val.jsonl"))

    print("Generating test set...")
    test = generate_dataset(args.n_test, seed=args.seed + 2)
    save_jsonl(test, os.path.join(args.output_dir, "test.jsonl"))

    from collections import Counter
    dist = Counter(s["task_type"] for s in train)
    print("\nTrain distribution:")
    for t, c in sorted(dist.items()):
        print(f"  {t}: {c}")
    print(f"\nTotal: {args.n_train + args.n_val + args.n_test} samples")


if __name__ == "__main__":
    main()
