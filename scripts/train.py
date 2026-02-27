"""
Training CLI for Duplex-1-1.7B.

Usage:
    python scripts/train.py --phase 1 --max_steps 10000
    python scripts/train.py --phase 2 --max_steps 20000 --resume checkpoints/duplex-1-1.7b/phase1_final.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duplex.config import DuplexConfig, TrainingConfig
from duplex.duplex_model import DuplexModel
from duplex.data.dataset import DuplexDataset
from duplex.training.trainer import DuplexTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Duplex-1-1.7B")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default="generated_data_duplex")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/duplex-1-1.7b")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--qwen_path", type=str, default="models/qwen3-1.7b-base")
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    duplex_config = DuplexConfig(
        qwen_model_path=args.qwen_path,
        quantize_4bit=not args.no_quantize,
    )

    train_config = TrainingConfig(
        phase=args.phase,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    print("Loading Duplex-1-1.7B model...")
    model = DuplexModel(duplex_config)

    print("\nLoading datasets...")
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run scripts/generate_data.py first.")
        sys.exit(1)

    train_ds = DuplexDataset.from_jsonl(
        train_path, model.tokenizer, phase=args.phase,
    )
    val_ds = DuplexDataset.from_jsonl(
        val_path, model.tokenizer, phase=args.phase,
    )

    trainer = DuplexTrainer(model, train_config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(train_ds, val_ds)


if __name__ == "__main__":
    main()
