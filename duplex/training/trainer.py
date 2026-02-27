"""
Trainer for Duplex-1: trains only the adapter components (encoder, workspace,
cross-attention adapters) while keeping Qwen3 frozen.
"""

import os
import time
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from duplex.duplex_model import DuplexModel
from duplex.data.dataset import DuplexDataset
from duplex.config import TrainingConfig


class DuplexTrainer:
    def __init__(self, model: DuplexModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = next(model.qwen.parameters()).device

        # Only optimize trainable (non-frozen) parameters
        trainable_params = model.get_trainable_params()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.global_step = 0
        self.log_history: list[dict] = []

        model.print_param_summary()

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

    def _train_step(self, batch: dict) -> dict[str, float]:
        self.model.train()

        out = self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            prompt_ids=batch["prompt_ids"].to(self.device),
            prompt_mask=batch["prompt_mask"].to(self.device),
            update_ids=batch["update_ids"].to(self.device),
            update_mask=batch["update_mask"].to(self.device),
            labels=batch["labels"].to(self.device),
        )

        loss = out["loss"] / self.config.gradient_accumulation_steps
        loss.backward()

        return {"loss": out["loss"].item()}

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Only save trainable components (not the full Qwen model)
        torch.save({
            "encoder_state_dict": self.model.encoder.state_dict(),
            "workspace_state_dict": self.model.workspace.state_dict(),
            "adapters_state_dict": self.model.adapters.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.model.workspace.load_state_dict(ckpt["workspace_state_dict"])
        self.model.adapters.load_state_dict(ckpt["adapters_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        old_step = ckpt["global_step"]
        old_phase = ckpt.get("config", self.config).phase

        if old_phase != self.config.phase:
            self.global_step = 0
            print(f"Loaded weights from step {old_step} (phase {old_phase}). "
                  f"Reset step counter for phase {self.config.phase}.")
        else:
            self.global_step = old_step
            print(f"Resumed from step {self.global_step}")

    @torch.no_grad()
    def _eval_loop(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            out = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                prompt_ids=batch["prompt_ids"].to(self.device),
                prompt_mask=batch["prompt_mask"].to(self.device),
                update_ids=batch["update_ids"].to(self.device),
                update_mask=batch["update_mask"].to(self.device),
                labels=batch["labels"].to(self.device),
            )
            total_loss += out["loss"].item()
            n_batches += 1

        return {"val_loss": total_loss / max(1, n_batches)}

    def train(self, train_dataset: DuplexDataset, val_dataset: DuplexDataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=DuplexDataset.collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=DuplexDataset.collate_fn,
            num_workers=0,
        )

        ckpt_dir = self.config.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        print(f"\nTraining Duplex-1-1.7B | Phase {self.config.phase}")
        print(f"Max steps: {self.config.max_steps} | Batch: {self.config.batch_size}")
        print(f"Grad accum: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

        t_start = time.time()
        running_loss = 0.0
        n_accumulated = 0
        accum_count = 0
        epoch = 0

        self.optimizer.zero_grad()

        while self.global_step < self.config.max_steps:
            epoch += 1
            for batch in train_loader:
                if self.global_step >= self.config.max_steps:
                    break

                lr = self._get_lr(self.global_step)
                self._set_lr(lr)

                metrics = self._train_step(batch)
                running_loss += metrics["loss"]
                n_accumulated += 1
                accum_count += 1

                if accum_count >= self.config.gradient_accumulation_steps:
                    nn.utils.clip_grad_norm_(
                        self.model.get_trainable_params(),
                        self.config.grad_clip,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accum_count = 0
                    self.global_step += 1

                    if self.global_step % self.config.log_every == 0:
                        avg_loss = running_loss / n_accumulated
                        elapsed = time.time() - t_start
                        steps_per_sec = self.global_step / elapsed
                        entry = {
                            "step": self.global_step,
                            "loss": avg_loss,
                            "lr": lr,
                            "steps_per_sec": steps_per_sec,
                        }
                        self.log_history.append(entry)
                        print(
                            f"Step {self.global_step:6d} | loss: {avg_loss:.4f} | "
                            f"lr: {lr:.2e} | {steps_per_sec:.2f} steps/s"
                        )
                        running_loss = 0.0
                        n_accumulated = 0

                    if self.global_step % self.config.eval_every == 0:
                        val_metrics = self._eval_loop(val_loader)
                        print(f"  >> Eval step {self.global_step}: val_loss={val_metrics['val_loss']:.4f}")
                        self.log_history.append({
                            "step": self.global_step,
                            "val_loss": val_metrics["val_loss"],
                        })

                    if self.global_step % self.config.save_every == 0:
                        path = os.path.join(ckpt_dir, f"step_{self.global_step}.pt")
                        self.save_checkpoint(path)
                        print(f"  >> Saved: {path}")

        final_path = os.path.join(ckpt_dir, "final.pt")
        self.save_checkpoint(final_path)
        print(f"\nTraining complete. Final: {final_path}")

        log_path = os.path.join(ckpt_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.log_history, f, indent=2)
