"""
Dataset for Duplex-1 training. Each sample provides:
    - prompt_ids / prompt_mask: for the encoder (init workspace)
    - update_ids / update_mask: for the encoder (correction, may be empty)
    - input_ids / attention_mask / labels: for the decoder (text to generate)
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DuplexDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_prompt_len: int = 128,
        max_response_len: int = 256,
        max_correction_len: int = 64,
        phase: int = 2,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.max_correction_len = max_correction_len
        self.phase = phase

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]

        # Encoder input: prompt
        prompt_enc = self.tokenizer(
            s["prompt"],
            max_length=self.max_prompt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Encoder input: correction (Phase 2 only)
        if self.phase == 2 and s.get("correction"):
            update_enc = self.tokenizer(
                s["correction"],
                max_length=self.max_correction_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            update_enc = self.tokenizer(
                "",
                max_length=self.max_correction_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        # Decoder target: the text the model should generate
        if self.phase == 1:
            target_text = s["partial_response"]
        else:
            target_text = s["revised_continuation"]

        # Decoder input = prompt + target (model sees prompt, predicts target)
        full_text = s["prompt"] + " " + target_text
        decoder_enc = self.tokenizer(
            full_text,
            max_length=self.max_response_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Labels: -100 on prompt tokens (don't compute loss there)
        prompt_only = self.tokenizer(
            s["prompt"] + " ",
            max_length=self.max_response_len,
            truncation=True,
        )
        prompt_token_len = len(prompt_only["input_ids"])

        labels = decoder_enc["input_ids"].clone().squeeze(0)
        labels[:prompt_token_len] = -100
        # Also mask padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "prompt_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_mask": prompt_enc["attention_mask"].squeeze(0),
            "update_ids": update_enc["input_ids"].squeeze(0),
            "update_mask": update_enc["attention_mask"].squeeze(0),
            "input_ids": decoder_enc["input_ids"].squeeze(0),
            "attention_mask": decoder_enc["attention_mask"].squeeze(0),
            "labels": labels,
            "task_type": s["task_type"],
            "expected_values": json.dumps(s.get("expected_values", {})),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
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
        tokenizer: PreTrainedTokenizer,
        max_prompt_len: int = 128,
        max_response_len: int = 256,
        max_correction_len: int = 64,
        phase: int = 2,
    ) -> "DuplexDataset":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return cls(samples, tokenizer, max_prompt_len, max_response_len, max_correction_len, phase)
