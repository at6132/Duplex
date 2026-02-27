"""
DuplexTrainer: trains only the adapter components (encoder, workspace,
cross-attention adapters) while keeping Qwen3 frozen.

Supports:
  - Single-GPU and multi-GPU DDP (via torchrun)
  - bfloat16 autocast for maximum H200 throughput
  - Pinned memory + prefetch DataLoader workers
  - Cosine LR schedule with linear warmup
  - Gradient accumulation + clipping
  - Only rank-0 saves checkpoints and prints logs
"""

import os
import signal
import time
import json
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast

from duplex.duplex_model import DuplexModel
from duplex.data.dataset import DuplexDataset
from duplex.config import TrainingConfig


class DuplexTrainer:
    def __init__(self, model: DuplexModel, config: TrainingConfig):
        self.config = config
        self.is_ddp = config.use_ddp
        self.local_rank = config.local_rank
        self.world_size = config.world_size
        self.is_main = (self.local_rank <= 0)

        # Wrap in DDP if using multiple GPUs
        if self.is_ddp:
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            self.raw_model = model
        else:
            self.model = model
            self.raw_model = model

        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        trainable_params = self.raw_model.get_trainable_params()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True,  # fused AdamW kernel — faster on H200
        )

        self.global_step = 0
        self.log_history: list[dict] = []
        self._stop_requested = False

        # Save on Ctrl+C / SIGTERM instead of dying with no checkpoint
        if self.is_main:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

        if self.is_main:
            self.raw_model.print_param_summary()

    def _handle_signal(self, signum, frame):
        if self.is_main:
            print(f"\nSignal {signum} received — saving checkpoint before exit...")
        self._stop_requested = True

    def _get_lr(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / max(1, self.config.warmup_steps)
        progress = (step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _train_step(self, batch: dict) -> float:
        self.model.train()

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model(
                input_ids=batch["input_ids"].to(self.device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(self.device, non_blocking=True),
                prompt_ids=batch["prompt_ids"].to(self.device, non_blocking=True),
                prompt_mask=batch["prompt_mask"].to(self.device, non_blocking=True),
                update_ids=batch["update_ids"].to(self.device, non_blocking=True),
                update_mask=batch["update_mask"].to(self.device, non_blocking=True),
                labels=batch["labels"].to(self.device, non_blocking=True),
            )

        loss = out["loss"] / self.config.gradient_accumulation_steps
        loss.backward()
        return out["loss"].item()

    @torch.no_grad()
    def _eval_loop(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in dataloader:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(
                    input_ids=batch["input_ids"].to(self.device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(self.device, non_blocking=True),
                    prompt_ids=batch["prompt_ids"].to(self.device, non_blocking=True),
                    prompt_mask=batch["prompt_mask"].to(self.device, non_blocking=True),
                    update_ids=batch["update_ids"].to(self.device, non_blocking=True),
                    update_mask=batch["update_mask"].to(self.device, non_blocking=True),
                    labels=batch["labels"].to(self.device, non_blocking=True),
                )
            total_loss += out["loss"].item()
            n += 1

        # Average across all DDP ranks
        avg = total_loss / max(1, n)
        if self.is_ddp:
            t = torch.tensor(avg, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = t.item()
        return avg

    def save_checkpoint(self, path: str):
        if not self.is_main:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "encoder_state_dict": self.raw_model.encoder.state_dict(),
            "workspace_state_dict": self.raw_model.workspace.state_dict(),
            "adapters_state_dict": self.raw_model.adapters.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.raw_model.workspace.load_state_dict(ckpt["workspace_state_dict"])
        self.raw_model.adapters.load_state_dict(ckpt["adapters_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        old_step = ckpt["global_step"]
        old_phase = ckpt.get("config", self.config).phase

        if old_phase != self.config.phase:
            self.global_step = 0
            if self.is_main:
                print(f"Loaded weights from step {old_step} (phase {old_phase}). "
                      f"Reset step counter for phase {self.config.phase}.")
        else:
            self.global_step = old_step
            if self.is_main:
                print(f"Resumed from step {self.global_step}")

    def train(self, train_dataset: DuplexDataset, val_dataset: DuplexDataset):
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.is_ddp else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.is_ddp else None

        n_workers = min(8, os.cpu_count() or 4)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=DuplexDataset.collate_fn,
            num_workers=n_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, self.config.batch_size // 4),
            sampler=val_sampler,
            shuffle=False,
            collate_fn=DuplexDataset.collate_fn,
            num_workers=n_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        ckpt_dir = self.config.checkpoint_dir
        if self.is_main:
            os.makedirs(ckpt_dir, exist_ok=True)
            eff_batch = (self.config.batch_size
                         * self.config.gradient_accumulation_steps
                         * self.world_size)
            print(f"\nTraining Duplex-1-1.7B | Phase {self.config.phase}")
            print(f"GPUs: {self.world_size} | Batch/GPU: {self.config.batch_size} | "
                  f"Grad accum: {self.config.gradient_accumulation_steps} | "
                  f"Effective batch: {eff_batch}")
            print(f"Max steps: {self.config.max_steps} | Warmup: {self.config.warmup_steps}")
            print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}\n")

        t_start = time.time()
        running_loss = 0.0
        n_accumulated = 0
        accum_count = 0
        epoch = 0

        # Convergence detection: stop if loss doesn't improve by min_delta
        # for patience consecutive log intervals
        best_loss = float("inf")
        patience = 10
        patience_counter = 0
        min_delta = 1e-4

        self.optimizer.zero_grad()

        while self.global_step < self.config.max_steps and not self._stop_requested:
            epoch += 1
            if self.is_ddp:
                train_sampler.set_epoch(epoch)

            for batch in train_loader:
                if self.global_step >= self.config.max_steps or self._stop_requested:
                    break

                self._set_lr(self._get_lr(self.global_step))
                loss_val = self._train_step(batch)
                running_loss += loss_val
                n_accumulated += 1
                accum_count += 1

                if accum_count >= self.config.gradient_accumulation_steps:
                    nn.utils.clip_grad_norm_(
                        self.raw_model.get_trainable_params(),
                        self.config.grad_clip,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accum_count = 0
                    self.global_step += 1

                    if self.is_main and self.global_step % self.config.log_every == 0:
                        avg_loss = running_loss / n_accumulated
                        elapsed = time.time() - t_start
                        steps_per_sec = self.global_step / elapsed
                        tokens_per_sec = (steps_per_sec
                                          * self.config.batch_size
                                          * self.config.max_seq_len
                                          * self.world_size)
                        lr = self._get_lr(self.global_step)
                        entry = {
                            "step": self.global_step,
                            "loss": avg_loss,
                            "lr": lr,
                            "steps_per_sec": steps_per_sec,
                            "tokens_per_sec": tokens_per_sec,
                        }
                        self.log_history.append(entry)
                        print(
                            f"Step {self.global_step:6d} | loss: {avg_loss:.4f} | "
                            f"lr: {lr:.2e} | {steps_per_sec:.2f} steps/s | "
                            f"{tokens_per_sec/1000:.1f}k tok/s"
                        )

                        # Convergence check
                        if avg_loss < best_loss - min_delta:
                            best_loss = avg_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"\nConverged — loss flat for {patience} log intervals. Stopping.")
                                self._stop_requested = True

                        running_loss = 0.0
                        n_accumulated = 0

                    if self.global_step % self.config.eval_every == 0:
                        val_loss = self._eval_loop(val_loader)
                        if self.is_main:
                            print(f"  >> Eval {self.global_step}: val_loss={val_loss:.4f}")
                            self.log_history.append({
                                "step": self.global_step,
                                "val_loss": val_loss,
                            })

                    if self.global_step % self.config.save_every == 0:
                        path = os.path.join(ckpt_dir, f"step_{self.global_step}.pt")
                        self.save_checkpoint(path)
                        if self.is_main:
                            print(f"  >> Saved: {path}")

        final_path = os.path.join(ckpt_dir, "final.pt")
        self.save_checkpoint(final_path)
        if self.is_main:
            print(f"\nTraining complete. Final: {final_path}")
            log_path = os.path.join(ckpt_dir, "training_log.json")
            with open(log_path, "w") as f:
                json.dump(self.log_history, f, indent=2)
