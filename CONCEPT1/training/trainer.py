import os
import time
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.tokenizer import CharTokenizer
from data.dataset import DuplexDataset
from models.baseline import BaselineDecoder
from models.workspace_generator import WorkspaceGenerator
from configs.base_config import ExperimentConfig


class Trainer:
    """Unified trainer for both baseline and workspace models."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tc = config.training
        self.mc = config.model
        self.dc = config.data

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CharTokenizer()

        self.model = self._build_model()
        self.model.to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {self.tc.model_type} | Parameters: {param_count:,} | Device: {self.device}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.tc.learning_rate,
            weight_decay=self.tc.weight_decay,
        )

        self.global_step = 0
        self.log_history: list[dict] = []

    def _build_model(self) -> nn.Module:
        if self.tc.model_type == "baseline":
            return BaselineDecoder(
                vocab_size=self.mc.vocab_size,
                d_model=self.mc.d_model,
                n_heads=self.mc.n_heads,
                d_ff=self.mc.d_ff,
                n_layers=self.mc.n_layers,
                max_seq_len=self.mc.max_seq_len,
                dropout=self.mc.dropout,
                pad_id=self.tokenizer.pad_id,
            )
        elif self.tc.model_type == "workspace":
            return WorkspaceGenerator(
                vocab_size=self.mc.vocab_size,
                d_model=self.mc.d_model,
                n_heads=self.mc.n_heads,
                d_ff=self.mc.d_ff,
                n_decoder_layers=self.mc.n_layers,
                n_encoder_layers=self.mc.n_encoder_layers,
                n_workspace_slots=self.mc.n_workspace_slots,
                max_seq_len=self.mc.max_seq_len,
                dropout=self.mc.dropout,
                pad_id=self.tokenizer.pad_id,
            )
        else:
            raise ValueError(f"Unknown model type: {self.tc.model_type}")

    def _get_lr(self, step: int) -> float:
        if step < self.tc.warmup_steps:
            return self.tc.learning_rate * step / max(1, self.tc.warmup_steps)
        progress = (step - self.tc.warmup_steps) / max(
            1, self.tc.max_steps - self.tc.warmup_steps
        )
        return self.tc.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _train_step_baseline(self, batch: dict) -> dict[str, float]:
        input_ids = batch["input_ids"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)
        targets = batch["targets"].to(self.device)

        out = self.model(input_ids, loss_mask=loss_mask, targets=targets)
        loss = out["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
        self.optimizer.step()

        return {"loss": loss.item()}

    def _train_step_workspace(self, batch: dict) -> dict[str, float]:
        prompt_ids = batch["prompt_ids"].to(self.device)
        prefix_ids = batch["prefix_ids"].to(self.device)
        update_ids = batch["update_ids"].to(self.device)
        continuation_ids = batch["continuation_ids"].to(self.device)
        prefix_loss_mask = batch["prefix_loss_mask"].to(self.device)
        continuation_loss_mask = batch["continuation_loss_mask"].to(self.device)

        out = self.model(
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
            update_ids=update_ids,
            continuation_ids=continuation_ids,
            prefix_loss_mask=prefix_loss_mask,
            continuation_loss_mask=continuation_loss_mask,
        )
        loss = out["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "workspace_delta": out["workspace_delta"].item(),
        }

    def _eval_loop(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if self.tc.model_type == "baseline":
                    input_ids = batch["input_ids"].to(self.device)
                    loss_mask = batch["loss_mask"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    out = self.model(input_ids, loss_mask=loss_mask, targets=targets)
                else:
                    out = self.model(
                        prompt_ids=batch["prompt_ids"].to(self.device),
                        prefix_ids=batch["prefix_ids"].to(self.device),
                        update_ids=batch["update_ids"].to(self.device),
                        continuation_ids=batch["continuation_ids"].to(self.device),
                        prefix_loss_mask=batch["prefix_loss_mask"].to(self.device),
                        continuation_loss_mask=batch["continuation_loss_mask"].to(self.device),
                    )
                total_loss += out["loss"].item()
                n_batches += 1

        self.model.train()
        return {"val_loss": total_loss / max(1, n_batches)}

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str, reset_steps: bool = False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        old_step = ckpt["global_step"]
        old_phase = ckpt.get("config", self.config).training.phase

        if reset_steps or old_phase != self.config.training.phase:
            self.global_step = 0
            print(f"Loaded weights from step {old_step} (phase {old_phase}). "
                  f"Reset step counter to 0 for phase {self.config.training.phase}.")
        else:
            self.global_step = old_step
            print(f"Resumed from step {self.global_step}")

    def train(
        self,
        train_dataset: DuplexDataset,
        val_dataset: DuplexDataset,
    ):
        mode = "baseline" if self.tc.model_type == "baseline" else "experimental"

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.tc.batch_size,
            shuffle=True,
            collate_fn=DuplexDataset.collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.tc.batch_size,
            shuffle=False,
            collate_fn=DuplexDataset.collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        ckpt_dir = os.path.join(
            self.tc.checkpoint_dir, self.config.experiment_name
        )
        os.makedirs(ckpt_dir, exist_ok=True)

        self.model.train()
        epoch = 0
        step_fn = (
            self._train_step_baseline
            if self.tc.model_type == "baseline"
            else self._train_step_workspace
        )

        print(f"\nTraining {self.tc.model_type} | Phase {self.tc.phase} | Max steps: {self.tc.max_steps}")
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}\n")

        t_start = time.time()
        running_loss = 0.0
        n_accumulated = 0

        while self.global_step < self.tc.max_steps:
            epoch += 1
            for batch in train_loader:
                if self.global_step >= self.tc.max_steps:
                    break

                lr = self._get_lr(self.global_step)
                self._set_lr(lr)

                metrics = step_fn(batch)
                running_loss += metrics["loss"]
                n_accumulated += 1
                self.global_step += 1

                if self.global_step % self.tc.log_every == 0:
                    avg_loss = running_loss / n_accumulated
                    elapsed = time.time() - t_start
                    steps_per_sec = self.global_step / elapsed
                    log_entry = {
                        "step": self.global_step,
                        "loss": avg_loss,
                        "lr": lr,
                        "steps_per_sec": steps_per_sec,
                        "epoch": epoch,
                    }
                    if "workspace_delta" in metrics:
                        log_entry["workspace_delta"] = metrics["workspace_delta"]

                    self.log_history.append(log_entry)
                    ws_str = (
                        f" | ws_delta: {metrics['workspace_delta']:.4f}"
                        if "workspace_delta" in metrics
                        else ""
                    )
                    print(
                        f"Step {self.global_step:6d} | loss: {avg_loss:.4f}"
                        f" | lr: {lr:.2e} | {steps_per_sec:.1f} steps/s{ws_str}"
                    )
                    running_loss = 0.0
                    n_accumulated = 0

                if self.global_step % self.tc.eval_every == 0:
                    val_metrics = self._eval_loop(val_loader)
                    print(f"  >> Eval at step {self.global_step}: val_loss = {val_metrics['val_loss']:.4f}")
                    self.log_history.append({
                        "step": self.global_step,
                        "val_loss": val_metrics["val_loss"],
                    })

                if self.global_step % self.tc.save_every == 0:
                    path = os.path.join(ckpt_dir, f"step_{self.global_step}.pt")
                    self.save_checkpoint(path)
                    print(f"  >> Saved checkpoint: {path}")

        # Final save
        final_path = os.path.join(ckpt_dir, "final.pt")
        self.save_checkpoint(final_path)
        print(f"\nTraining complete. Final checkpoint: {final_path}")

        # Save log history
        log_path = os.path.join(ckpt_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.log_history, f, indent=2)
        print(f"Training log saved: {log_path}")
