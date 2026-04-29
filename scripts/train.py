"""
Training CLI for Duplex-1.5-4B (Gemma 3 4B local).

Single GPU (local):
    python scripts/train.py --phase 1 --max_steps 5000

Resume Phase 2:
    python scripts/train.py --phase 2 --max_steps 5000 --resume checkpoints/duplex-1.5-4b/phase1_best.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duplex.config import DuplexConfig, TrainingConfig
from duplex.duplex_model import DuplexModel
from duplex.data.dataset import DuplexDataset
from duplex.training.trainer import DuplexTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Duplex-1.5-4B")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default="generated_data_duplex")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    duplex_config = DuplexConfig()
    if args.model_path:
        duplex_config.model_path = args.model_path

    train_config = TrainingConfig(
        phase=args.phase,
        seed=args.seed,
    )
    if args.checkpoint_dir:
        train_config.checkpoint_dir = args.checkpoint_dir
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.grad_accum is not None:
        train_config.gradient_accumulation_steps = args.grad_accum
    if args.learning_rate is not None:
        train_config.learning_rate = args.learning_rate

    # Auto-adjust warmup: 10% of max_steps, capped at configured default
    auto_warmup = max(50, train_config.max_steps // 10)
    train_config.warmup_steps = min(train_config.warmup_steps, auto_warmup)

    print("Loading Duplex-1.5-4B (Gemma 3 4B-IT)...")
    model = DuplexModel(duplex_config)

    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run scripts/generate_data.py first.")
        sys.exit(1)

    print("Loading datasets...")
    train_ds = DuplexDataset.from_jsonl(train_path, model.tokenizer, phase=args.phase)
    val_ds = DuplexDataset.from_jsonl(val_path, model.tokenizer, phase=args.phase)

    trainer = DuplexTrainer(model, train_config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(train_ds, val_ds)


if __name__ == "__main__":
    main()
