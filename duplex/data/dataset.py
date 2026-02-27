"""
Dataset for Duplex-1 training.

CRITICAL DESIGN:
  Phase 1: Decoder sees prompt, predicts partial_response. Workspace = encode(prompt).
           This teaches the adapters to condition on the workspace.

  Phase 2: Decoder sees prompt + WRONG partial_response (cut mid-sentence),
           but labels = REVISED continuation. Workspace = encode(prompt) then
           gated_update(correction). The model MUST read the workspace to know
           what changed — the correction info is NOT in the decoder input.

This eliminates the shortcut where the model ignores the workspace because
the answer was already in the decoder input.
"""

import json
import random
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DuplexDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_prompt_len: int = 128,
        max_response_len: int = 384,
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

        # --- Encoder: prompt ---
        prompt_enc = self.tokenizer(
            s["prompt"],
            max_length=self.max_prompt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if self.phase == 1:
            # Phase 1: no correction, predict partial_response from prompt
            update_enc = self.tokenizer(
                "",
                max_length=self.max_correction_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # Decoder: prompt → partial_response
            full_text = s["prompt"] + " " + s["partial_response"]
            target_start_text = s["prompt"] + " "
        else:
            # Phase 2: correction present, decoder sees WRONG context but must predict REVISED
            update_enc = self.tokenizer(
                s.get("correction", ""),
                max_length=self.max_correction_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Decoder input = prompt + partial_response (the WRONG answer)
            # Labels = only the revised_continuation part (the CORRECT answer)
            # This forces the model to read the workspace to know what changed.
            partial = s.get("partial_response", "")
            revised = s.get("revised_continuation", "")

            # Cut partial_response at a random point (30-80%) to simulate mid-generation
            if partial:
                words = partial.split()
                cut_frac = random.uniform(0.3, 0.8)
                cut_point = max(1, int(len(words) * cut_frac))
                partial_cut = " ".join(words[:cut_point])
            else:
                partial_cut = ""

            # Decoder sees: prompt + wrong_partial (cut) + revised_continuation
            # But loss is ONLY on revised_continuation
            full_text = s["prompt"] + " " + partial_cut + " " + revised
            target_start_text = s["prompt"] + " " + partial_cut + " "

        decoder_enc = self.tokenizer(
            full_text,
            max_length=self.max_response_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Mask everything up to the target start (only compute loss on the target)
        prefix_enc = self.tokenizer(
            target_start_text,
            max_length=self.max_response_len,
            truncation=True,
        )
        prefix_len = len(prefix_enc["input_ids"])

        labels = decoder_enc["input_ids"].clone().squeeze(0)
        labels[:prefix_len] = -100
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
        max_response_len: int = 384,
        max_correction_len: int = 64,
        phase: int = 2,
    ) -> "DuplexDataset":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return cls(samples, tokenizer, max_prompt_len, max_response_len, max_correction_len, phase)
