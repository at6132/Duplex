"""
Benchmark runner for evaluating and comparing baseline vs workspace models
on the duplex interruption/revision test set.
"""

import json
import os
import torch
from tqdm import tqdm

from data.tokenizer import CharTokenizer
from data.dataset import DuplexDataset
from models.baseline import BaselineDecoder
from models.workspace_generator import WorkspaceGenerator
from configs.base_config import ModelConfig
from eval.metrics import compute_metrics, AggregateMetrics


class Benchmark:
    """Runs generation + evaluation for baseline and workspace models."""

    def __init__(
        self,
        tokenizer: CharTokenizer | None = None,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ):
        self.tokenizer = tokenizer or CharTokenizer()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, checkpoint_path: str):
        """Load a model from checkpoint, auto-detecting type."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = ckpt["config"]
        mc = config.model

        if config.training.model_type == "baseline":
            model = BaselineDecoder(
                vocab_size=mc.vocab_size,
                d_model=mc.d_model,
                n_heads=mc.n_heads,
                d_ff=mc.d_ff,
                n_layers=mc.n_layers,
                max_seq_len=mc.max_seq_len,
                dropout=0.0,
                pad_id=self.tokenizer.pad_id,
            )
        else:
            model = WorkspaceGenerator(
                vocab_size=mc.vocab_size,
                d_model=mc.d_model,
                n_heads=mc.n_heads,
                d_ff=mc.d_ff,
                n_decoder_layers=mc.n_layers,
                n_encoder_layers=mc.n_encoder_layers,
                n_workspace_slots=mc.n_workspace_slots,
                max_seq_len=mc.max_seq_len,
                dropout=0.0,
                pad_id=self.tokenizer.pad_id,
            )

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model, config.training.model_type

    def evaluate_baseline(
        self,
        model: BaselineDecoder,
        test_samples: list[dict],
    ) -> tuple[AggregateMetrics, list[dict]]:
        """Generate and evaluate baseline model on test samples."""
        results = []
        raw_outputs = []

        for sample in tqdm(test_samples, desc="Evaluating baseline"):
            # Build prompt up to the UPDATE marker, then let model continue
            prompt_text = f"<BOS>{sample['prompt']}<SEP>{sample['output_prefix']}<UPDATE>{sample['update']}<SEP>"
            prompt_ids = self.tokenizer.encode(prompt_text)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

            generated = model.generate(
                prompt_tensor,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                eos_id=self.tokenizer.eos_id,
            )

            full_text = self.tokenizer.decode(generated[0].tolist())
            # Extract continuation (after the last <SEP>)
            parts = full_text.split("<SEP>")
            continuation = parts[-1] if len(parts) > 1 else full_text

            results.append((continuation, sample["expected_values"], sample["task_type"]))
            raw_outputs.append({
                "task_type": sample["task_type"],
                "prompt": sample["prompt"],
                "update": sample["update"],
                "expected": sample["expected_values"],
                "continuation": continuation,
                "full_output": full_text,
            })

        metrics = compute_metrics(results)
        return metrics, raw_outputs

    def evaluate_workspace(
        self,
        model: WorkspaceGenerator,
        test_samples: list[dict],
        update_after_step: int = 5,
    ) -> tuple[AggregateMetrics, list[dict]]:
        """Generate and evaluate workspace model on test samples."""
        results = []
        raw_outputs = []

        for sample in tqdm(test_samples, desc="Evaluating workspace"):
            prompt_ids = self.tokenizer.encode(sample["prompt"])
            update_ids = self.tokenizer.encode(sample["update"])

            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            update_tensor = torch.tensor([update_ids], dtype=torch.long, device=self.device)

            generated, ws_snapshots = model.generate(
                prompt_tensor,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                eos_id=self.tokenizer.eos_id,
                update_ids=update_tensor,
                update_after_step=update_after_step,
            )

            continuation = self.tokenizer.decode(generated[0].tolist())
            # Strip BOS if present
            continuation = continuation.replace("<BOS>", "").replace("<EOS>", "")

            results.append((continuation, sample["expected_values"], sample["task_type"]))
            raw_outputs.append({
                "task_type": sample["task_type"],
                "prompt": sample["prompt"],
                "update": sample["update"],
                "expected": sample["expected_values"],
                "continuation": continuation,
                "n_workspace_updates": len(ws_snapshots),
            })

        metrics = compute_metrics(results)
        return metrics, raw_outputs

    def compare(
        self,
        baseline_path: str,
        workspace_path: str,
        test_samples: list[dict],
        output_dir: str = "eval_results",
    ) -> dict:
        """Run both models and produce a comparison report."""
        os.makedirs(output_dir, exist_ok=True)

        print("\n=== Loading baseline model ===")
        baseline_model, _ = self.load_model(baseline_path)
        print("=== Loading workspace model ===")
        workspace_model, _ = self.load_model(workspace_path)

        print(f"\n=== Evaluating on {len(test_samples)} test samples ===\n")

        print("--- Baseline ---")
        baseline_metrics, baseline_raw = self.evaluate_baseline(baseline_model, test_samples)
        print("--- Workspace ---")
        workspace_metrics, workspace_raw = self.evaluate_workspace(workspace_model, test_samples)

        report = self._format_report(baseline_metrics, workspace_metrics)

        # Save results
        with open(os.path.join(output_dir, "baseline_outputs.json"), "w") as f:
            json.dump(baseline_raw, f, indent=2)
        with open(os.path.join(output_dir, "workspace_outputs.json"), "w") as f:
            json.dump(workspace_raw, f, indent=2)
        with open(os.path.join(output_dir, "comparison_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(output_dir, "comparison_report.txt"), "w") as f:
            f.write(self._format_report_text(baseline_metrics, workspace_metrics))

        print(self._format_report_text(baseline_metrics, workspace_metrics))

        return report

    def _format_report(
        self,
        baseline: AggregateMetrics,
        workspace: AggregateMetrics,
    ) -> dict:
        return {
            "baseline": {
                "revision_accuracy": baseline.revision_accuracy,
                "contradiction_rate": baseline.contradiction_rate,
                "mean_update_latency": baseline.mean_update_latency,
                "n_samples": baseline.n_samples,
                "per_task": baseline.per_task,
            },
            "workspace": {
                "revision_accuracy": workspace.revision_accuracy,
                "contradiction_rate": workspace.contradiction_rate,
                "mean_update_latency": workspace.mean_update_latency,
                "n_samples": workspace.n_samples,
                "per_task": workspace.per_task,
            },
        }

    def _format_report_text(
        self,
        baseline: AggregateMetrics,
        workspace: AggregateMetrics,
    ) -> str:
        lines = [
            "",
            "=" * 60,
            "  DUPLEX LM EVALUATION REPORT",
            "=" * 60,
            "",
            f"{'Metric':<30} {'Baseline':>12} {'Workspace':>12}",
            "-" * 60,
            f"{'Revision Accuracy':<30} {baseline.revision_accuracy:>11.1%} {workspace.revision_accuracy:>11.1%}",
            f"{'Contradiction Rate':<30} {baseline.contradiction_rate:>11.1%} {workspace.contradiction_rate:>11.1%}",
            f"{'Mean Update Latency (chars)':<30} {baseline.mean_update_latency:>12.1f} {workspace.mean_update_latency:>12.1f}",
            f"{'N Samples':<30} {baseline.n_samples:>12d} {workspace.n_samples:>12d}",
            "",
            "Per-Task Revision Accuracy:",
            "-" * 60,
        ]

        all_tasks = sorted(set(list(baseline.per_task.keys()) + list(workspace.per_task.keys())))
        for task in all_tasks:
            b_acc = baseline.per_task.get(task, {}).get("revision_accuracy", 0)
            w_acc = workspace.per_task.get(task, {}).get("revision_accuracy", 0)
            lines.append(f"  {task:<28} {b_acc:>11.1%} {w_acc:>11.1%}")

        lines.extend(["", "=" * 60, ""])
        return "\n".join(lines)
