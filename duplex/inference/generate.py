"""
Generation utilities for Duplex-1 and vanilla Qwen baseline comparison.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from duplex.duplex_model import DuplexModel
from duplex.config import DuplexConfig


def load_vanilla_qwen(model_path: str, quantize: bool = True):
    """Load vanilla Qwen3-1.7B for baseline comparison."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if quantize:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_duplex_model(
    qwen_path: str,
    checkpoint_path: str,
    quantize: bool = True,
) -> DuplexModel:
    """Load a trained Duplex-1-1.7B model."""
    config = DuplexConfig(qwen_model_path=qwen_path, quantize_4bit=quantize)
    model = DuplexModel(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.workspace.load_state_dict(ckpt["workspace_state_dict"])
    model.adapters.load_state_dict(ckpt["adapters_state_dict"])

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
    """Standard autoregressive generation with vanilla Qwen."""
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
    """
    Simulate baseline behavior: generate, then stop, re-prompt with correction, restart.

    Returns:
        (initial_partial_output, restarted_output_after_correction)
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate partial output
    partial_out = model.generate(
        **enc,
        max_new_tokens=tokens_before_correction,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    partial_text = tokenizer.decode(partial_out[0], skip_special_tokens=True)

    # Restart with correction appended to prompt
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
