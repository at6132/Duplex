"""
CLI entry point for training baseline or workspace models.

Usage:
    python scripts/train.py --model_type baseline --phase 1 --max_steps 50000
    python scripts/train.py --model_type workspace --phase 2 --max_steps 50000
    python scripts/train.py --model_type baseline --phase 1 --max_steps 500 --batch_size 16  # quick test
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig
from data.tokenizer import CharTokenizer
from data.dataset import DuplexDataset
from training.trainer import Trainer


def load_datasets(config: ExperimentConfig, tokenizer: CharTokenizer):
    data_dir = config.data.data_dir
    mode = "baseline" if config.training.model_type == "baseline" else "experimental"

    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run scripts/generate_data.py first.")
        sys.exit(1)

    max_seg = 128
    max_seq = config.model.max_seq_len

    train_ds = DuplexDataset.from_jsonl(train_path, tokenizer, mode, max_seq, max_seg)
    val_ds = DuplexDataset.from_jsonl(val_path, tokenizer, mode, max_seq, max_seg)

    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Train a duplex LM model")
    parser.add_argument("--model_type", type=str, default="baseline",
                        choices=["baseline", "workspace"])
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--data_dir", type=str, default="generated_data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = CharTokenizer()

    config = ExperimentConfig(
        model=ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        ),
        data=DataConfig(data_dir=args.data_dir),
        training=TrainingConfig(
            model_type=args.model_type,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            checkpoint_dir=args.checkpoint_dir,
            phase=args.phase,
            seed=args.seed,
        ),
    )

    train_ds, val_ds = load_datasets(config, tokenizer)

    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(train_ds, val_ds)


if __name__ == "__main__":
    main()
