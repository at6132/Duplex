"""
Dataset for Duplex-1.4 training (deep prefix conditioning / P-Tuning v2).

Key design: the prompt is ONLY available through the prefix (workspace).
The decoder input contains only a generic instruction + the response.
This forces the model to attend to the prefix for task-specific details.

CRITICAL: Token dropout (default 50%) randomly replaces response tokens in the
decoder input with pad tokens. Without this, teacher forcing lets the model
predict next tokens from text context alone, making the prefix redundant.
Token dropout forces the model to rely on the prefix for task-specific info.

Phase 1: Prefix = encode(prompt). Decoder sees generic instruction + corrupted response.
         Labels = original response tokens. Model MUST use prefix.
Phase 2: Prefix = encode(prompt) + encode(correction). Decoder sees generic
         instruction + corrupted partial + revised. Labels = revised part.
"""

import json
import random
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

GENERIC_INSTRUCTIONS = [
    "Respond to the following request.",
    "Answer the question below.",
    "Complete the task as instructed.",
    "Write a response based on the given context.",
    "Provide a helpful answer.",
    "Follow the instructions and respond.",
    "Generate a response for this task.",
    "Answer based on the provided information.",
]


class DuplexDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_prompt_len: int = 128,
        max_response_len: int = 512,
        max_correction_len: int = 96,
        phase: int = 2,
        token_dropout: float = 0.5,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.max_correction_len = max_correction_len
        self.phase = phase
        self.token_dropout = token_dropout
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]

        prompt_enc = self.tokenizer(
            s["prompt"],
            max_length=self.max_prompt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        generic = random.choice(GENERIC_INSTRUCTIONS)

        # Joint training: in Phase 2, randomly treat 40% of samples as Phase 1
        # (prompt-only, no correction). This prevents catastrophic forgetting of
        # prompt conditioning learned in Phase 1.
        effective_phase = self.phase
        if self.phase == 2 and random.random() < 0.4:
            effective_phase = 1

        if effective_phase == 1:
            update_enc = self.tokenizer(
                "",
                max_length=self.max_correction_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            response = s["partial_response"]
            full_text = generic + " " + response
            non_target = generic + " "
            prefix_len = len(self.tokenizer(non_target, add_special_tokens=True)["input_ids"])
        else:
            update_enc = self.tokenizer(
                s.get("correction", ""),
                max_length=self.max_correction_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            partial = s.get("partial_response", "")
            revised = s.get("revised_continuation", "")

            if partial:
                words = partial.split()
                cut_frac = random.uniform(0.3, 0.8)
                cut_point = max(1, int(len(words) * cut_frac))
                partial_cut = " ".join(words[:cut_point])
            else:
                partial_cut = ""

            full_text = generic + " " + partial_cut + " " + revised
            non_target = generic + " " + partial_cut + " "
            non_target_ids = self.tokenizer(non_target, add_special_tokens=True)["input_ids"]
            prefix_len = len(non_target_ids)

        decoder_enc = self.tokenizer(
            full_text,
            max_length=self.max_response_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Labels: original tokens (before dropout)
        labels = decoder_enc["input_ids"].clone().squeeze(0)
        labels[:prefix_len] = -100
        labels[labels == self.pad_token_id] = -100

        # Token dropout: randomly replace response tokens in decoder input with pad.
        # The generic instruction prefix is never corrupted — only the response.
        # This prevents the model from "cheating" with teacher-forced text context
        # and forces it to rely on the workspace prefix for task-specific info.
        input_ids = decoder_enc["input_ids"].squeeze(0).clone()
        if self.token_dropout > 0:
            seq_len = input_ids.size(0)
            dropout_mask = torch.rand(seq_len) < self.token_dropout
            dropout_mask[:prefix_len] = False  # protect generic instruction
            dropout_mask[input_ids == self.pad_token_id] = False  # don't corrupt padding
            input_ids[dropout_mask] = self.pad_token_id

        return {
            "prompt_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_mask": prompt_enc["attention_mask"].squeeze(0),
            "update_ids": update_enc["input_ids"].squeeze(0),
            "update_mask": update_enc["attention_mask"].squeeze(0),
            "input_ids": input_ids,
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
        max_response_len: int = 512,
        max_correction_len: int = 96,
        phase: int = 2,
        token_dropout: float = 0.5,
    ) -> "DuplexDataset":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return cls(samples, tokenizer, max_prompt_len, max_response_len, max_correction_len, phase, token_dropout)
