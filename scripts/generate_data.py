"""Generate synthetic training data for Duplex-1.

Usage:
    python scripts/generate_data.py --n_train 500000 --n_val 20000 --n_test 10000
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duplex.data.tasks import generate_dataset, TASK_GENERATORS


def _generate_chunk(args):
    n, seed = args
    return generate_dataset(n, seed=seed)


def parallel_generate(n_total: int, base_seed: int, n_workers: int) -> list[dict]:
    """Split n_total across n_workers processes and merge results."""
    chunk_size = n_total // n_workers
    chunks = [(chunk_size, base_seed + i) for i in range(n_workers)]
    # Give the last worker the remainder
    remainder = n_total - chunk_size * n_workers
    if remainder:
        last = chunks[-1]
        chunks[-1] = (last[0] + remainder, last[1])

    with Pool(n_workers) as pool:
        results = pool.map(_generate_chunk, chunks)

    return [sample for chunk in results for sample in chunk]


def save_jsonl(samples: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples):,} samples → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="generated_data_duplex")
    parser.add_argument("--n_train", type=int, default=500000)
    parser.add_argument("--n_val", type=int, default=20000)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: all CPU cores)"
    )
    args = parser.parse_args()

    n_workers = args.workers or cpu_count()
    print(f"Task types: {list(TASK_GENERATORS.keys())}")
    print(f"Workers:    {n_workers}")

    print(f"\nGenerating train set ({args.n_train:,} samples)...")
    train = parallel_generate(args.n_train, args.seed, n_workers)
    save_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))

    print(f"Generating val set ({args.n_val:,} samples)...")
    val = parallel_generate(args.n_val, args.seed + 1000, n_workers)
    save_jsonl(val, os.path.join(args.output_dir, "val.jsonl"))

    print(f"Generating test set ({args.n_test:,} samples)...")
    test = parallel_generate(args.n_test, args.seed + 2000, n_workers)
    save_jsonl(test, os.path.join(args.output_dir, "test.jsonl"))

    from collections import Counter
    dist = Counter(s["task_type"] for s in train)
    print("\nTrain distribution:")
    for t, c in sorted(dist.items()):
        print(f"  {t}: {c:,}")
    print(f"\nTotal: {args.n_train + args.n_val + args.n_test:,} samples")


if __name__ == "__main__":
    main()
