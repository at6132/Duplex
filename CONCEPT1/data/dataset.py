import json
import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


class DuplexDataset(Dataset):
    """
    PyTorch Dataset for duplex LM experiments.

    Supports two modes:
        - "baseline": serializes all fields into one flat token sequence
          with special markers (<BOS>, <SEP>, <UPDATE>, <EOS>).
        - "experimental": returns separate padded tensors for each segment
          (prompt, prefix, update, continuation).
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer: CharTokenizer,
        mode: str = "baseline",
        max_seq_len: int = 512,
        max_segment_len: int = 128,
    ):
        assert mode in ("baseline", "experimental")
        self.samples = samples
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.max_segment_len = max_segment_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        if self.mode == "baseline":
            return self._encode_baseline(s)
        return self._encode_experimental(s)

    def _encode_baseline(self, s: dict) -> dict[str, torch.Tensor]:
        """Serialize into a single sequence with loss mask."""
        enc = self.tokenizer.encode_with_special(
            s["prompt"], s["output_prefix"], s["update"], s["revised_continuation"]
        )
        ids = enc["input_ids"][:self.max_seq_len]
        prompt_len = min(enc["prompt_len"], len(ids))

        # Loss mask: 0 on prompt region, 1 on generation region
        loss_mask = [0] * prompt_len + [1] * (len(ids) - prompt_len)

        ids = self._pad(ids, self.max_seq_len)
        loss_mask = self._pad(loss_mask, self.max_seq_len)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
            "targets": torch.tensor(ids, dtype=torch.long),
            "task_type": s["task_type"],
            "expected_values": s["expected_values"],
        }

    def _encode_experimental(self, s: dict) -> dict[str, torch.Tensor]:
        """Return separate padded tensors for each segment."""
        enc = self.tokenizer.encode_structured(
            s["prompt"], s["output_prefix"], s["update"], s["revised_continuation"]
        )

        prompt_ids = enc["prompt_ids"][:self.max_segment_len]
        prefix_ids = enc["prefix_ids"][:self.max_segment_len]
        update_ids = enc["update_ids"][:self.max_segment_len]
        continuation_ids = enc["continuation_ids"][:self.max_segment_len]

        # Loss masks: all 1s for output segments (loss on every predicted token)
        prefix_mask = [1.0] * len(prefix_ids)
        continuation_mask = [1.0] * len(continuation_ids)

        return {
            "prompt_ids": torch.tensor(
                self._pad(prompt_ids, self.max_segment_len), dtype=torch.long
            ),
            "prefix_ids": torch.tensor(
                self._pad(prefix_ids, self.max_segment_len), dtype=torch.long
            ),
            "update_ids": torch.tensor(
                self._pad(update_ids, self.max_segment_len), dtype=torch.long
            ),
            "continuation_ids": torch.tensor(
                self._pad(continuation_ids, self.max_segment_len), dtype=torch.long
            ),
            "prefix_loss_mask": torch.tensor(
                self._pad(prefix_mask, self.max_segment_len), dtype=torch.float
            ),
            "continuation_loss_mask": torch.tensor(
                self._pad(continuation_mask, self.max_segment_len), dtype=torch.float
            ),
            "task_type": s["task_type"],
            "expected_values": s["expected_values"],
        }

    def _pad(self, seq: list, length: int) -> list:
        if len(seq) >= length:
            return seq[:length]
        return seq + [0] * (length - len(seq))

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Custom collate that handles the non-tensor metadata fields."""
        result = {}
        tensor_keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
        meta_keys = [k for k in batch[0] if k not in tensor_keys]

        for k in tensor_keys:
            result[k] = torch.stack([b[k] for b in batch])

        for k in meta_keys:
            result[k] = [b[k] for b in batch]

        return result

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer: CharTokenizer,
        mode: str = "baseline",
        max_seq_len: int = 512,
        max_segment_len: int = 128,
    ) -> "DuplexDataset":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return cls(samples, tokenizer, mode, max_seq_len, max_segment_len)
