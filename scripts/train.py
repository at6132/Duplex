"""
Training CLI for Duplex-1-1.7B v2.

Single GPU:
    python scripts/train.py --phase 1

Multi-GPU (both H200s):
    torchrun --nproc_per_node=2 scripts/train.py --phase 1

Resume Phase 2:
    torchrun --nproc_per_node=2 scripts/train.py --phase 2 --resume checkpoints/duplex-1-1.7b/phase1_best.pt
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duplex.config import DuplexConfig, TrainingConfig
from duplex.duplex_model import DuplexModel
from duplex.data.dataset import DuplexDataset
from duplex.training.trainer import DuplexTrainer


def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return False, 0, 1
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, local_rank, dist.get_world_size()


def main():
    is_ddp, local_rank, world_size = setup_ddp()
    is_main = local_rank <= 0

    parser = argparse.ArgumentParser(description="Train Duplex-1-1.7B v2")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default="generated_data_duplex")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/duplex-1-1.7b")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--qwen_path", type=str, default="models/qwen3-1.7b-base")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed + local_rank)

    duplex_config = DuplexConfig(
        qwen_model_path=args.qwen_path,
        quantize_4bit=False,
    )

    train_config = TrainingConfig(
        phase=args.phase,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        use_ddp=is_ddp,
        world_size=world_size,
        local_rank=local_rank,
    )
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.grad_accum is not None:
        train_config.gradient_accumulation_steps = args.grad_accum
    if args.learning_rate is not None:
        train_config.learning_rate = args.learning_rate

    if is_main:
        print("Loading Duplex-1-1.7B v2...")
    model = DuplexModel(duplex_config, local_rank=local_rank)

    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run scripts/generate_data.py first.")
        sys.exit(1)

    if is_main:
        print("Loading datasets...")
    train_ds = DuplexDataset.from_jsonl(train_path, model.tokenizer, phase=args.phase)
    val_ds = DuplexDataset.from_jsonl(val_path, model.tokenizer, phase=args.phase)

    trainer = DuplexTrainer(model, train_config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(train_ds, val_ds)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
