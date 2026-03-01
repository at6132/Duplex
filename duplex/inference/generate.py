"""
Generation utilities for Duplex-1 v2 and vanilla Qwen baseline comparison.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from duplex.duplex_model import DuplexModel
from duplex.config import DuplexConfig


def load_vanilla_qwen(model_path: str, quantize: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_duplex_model(
    qwen_path: str,
    checkpoint_path: str,
) -> DuplexModel:
    config = DuplexConfig(qwen_model_path=qwen_path, quantize_4bit=False)
    model = DuplexModel(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
    model.workspace.load_state_dict(ckpt["workspace_state_dict"], strict=False)
    if "embed_weight" in ckpt:
        model.qwen.model.embed_tokens.weight.data.copy_(ckpt["embed_weight"])
    if "lm_head_weight" in ckpt:
        model.qwen.lm_head.weight.data.copy_(ckpt["lm_head_weight"])

    model.eval()
    return model


@torch.no_grad()
def generate_vanilla(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


@torch.no_grad()
def generate_vanilla_with_restart(
    model,
    tokenizer,
    prompt: str,
    correction: str,
    tokens_before_correction: int = 30,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> tuple[str, str]:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    partial_out = model.generate(
        **enc,
        max_new_tokens=tokens_before_correction,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    partial_text = tokenizer.decode(partial_out[0], skip_special_tokens=True)

    corrected_prompt = f"{prompt}\n\n[User correction: {correction}]\n\nPlease respond with the correction in mind:\n"
    enc2 = tokenizer(corrected_prompt, return_tensors="pt").to(model.device)
    restarted_out = model.generate(
        **enc2,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    restarted_text = tokenizer.decode(restarted_out[0], skip_special_tokens=True)

    return partial_text, restarted_text


BASELINE_STOP_MSG = (
    "\n\n"
    "=== STOPPED === Correction received === Restarting from scratch ===\n\n"
)


def generate_vanilla_with_restart_streaming(
    model,
    tokenizer,
    prompt: str,
    correction: str,
    tokens_before_correction: int = 30,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]

    for _ in range(tokens_before_correction):
        out = model.generate(
            input_ids,
            max_new_tokens=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_ids = out
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        yield text

    text_after_partial = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    yield text_after_partial + BASELINE_STOP_MSG

    corrected_prompt = f"{prompt}\n\n[User correction: {correction}]\n\nPlease respond with the correction in mind:\n"
    enc2 = tokenizer(corrected_prompt, return_tensors="pt").to(device)
    input_ids = enc2["input_ids"]
    prompt_len = input_ids.shape[1]

    for _ in range(max_new_tokens - 1):
        out = model.generate(
            input_ids,
            max_new_tokens=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_ids = out
        restarted_only = tokenizer.decode(input_ids[0][prompt_len:], skip_special_tokens=True)
        combined = text_after_partial + BASELINE_STOP_MSG + restarted_only
        yield combined
        if out[0, -1].item() == tokenizer.eos_token_id:
            break
