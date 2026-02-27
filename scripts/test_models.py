"""Quick CLI test: run a few scenarios through both models and print results."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from duplex.inference.generate import load_vanilla_qwen, load_duplex_model, generate_vanilla

SYSTEM = (
    "You are a helpful assistant. Answer the user's question directly and concisely. "
    "When you receive new information or a correction mid-response, briefly acknowledge it "
    "and then continue with the updated information. Do not generate any extra conversation turns.\n\n"
)

SCENARIOS = [
    {
        "name": "Fact correction",
        "prompt": "What is the capital of Australia? Explain briefly.",
        "correction": "That's wrong — the capital is Canberra, not Sydney.",
        "inject_after": 12,
    },
    {
        "name": "Math correction",
        "prompt": "Given x = 5, compute 3*x + 7 step by step.",
        "correction": "Actually, x = 12 not 5. Please recalculate.",
        "inject_after": 10,
    },
    {
        "name": "Topic switch",
        "prompt": "Explain how photosynthesis works in plants.",
        "correction": "Switch to explaining cellular respiration instead.",
        "inject_after": 15,
    },
    {
        "name": "Profile update",
        "prompt": "Create a profile for Alice: age 25, software engineer in New York.",
        "correction": "Update: Alice is 32 and works as a doctor in London.",
        "inject_after": 12,
    },
]


def main():
    qwen_path = "models/qwen3-1.7b-base"
    ckpt_path = "checkpoints/duplex-1-1.7b/final.pt"

    if not os.path.exists(qwen_path):
        print(f"ERROR: {qwen_path} not found. Download Qwen first.")
        return
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found.")
        return

    print("Loading vanilla Qwen...")
    qwen_model, qwen_tok = load_vanilla_qwen(qwen_path, quantize=False)

    print("Loading Duplex...")
    duplex_model = load_duplex_model(qwen_path, ckpt_path)
    print("Both loaded.\n")

    for s in SCENARIOS:
        prompt = SYSTEM + s["prompt"]
        correction = s["correction"]
        inject = s["inject_after"]

        print("=" * 70)
        print(f"SCENARIO: {s['name']}")
        print(f"  Prompt:     {s['prompt']}")
        print(f"  Correction: {correction}")
        print(f"  Inject at:  token {inject}")
        print("=" * 70)

        # Vanilla: just answer the prompt (no correction — baseline behavior)
        vanilla_out = generate_vanilla(qwen_model, qwen_tok, prompt, max_new_tokens=120, temperature=0.5)
        # Strip system prompt from output
        if vanilla_out.startswith(SYSTEM):
            vanilla_out = vanilla_out[len(SYSTEM):]

        print(f"\n[VANILLA — no correction]\n{vanilla_out.strip()}")

        # Vanilla: answer with correction baked into prompt (restart simulation)
        restart_prompt = f"{prompt}\n\n[User correction: {correction}]\n\nPlease respond with the correction in mind:\n"
        vanilla_restart = generate_vanilla(qwen_model, qwen_tok, restart_prompt, max_new_tokens=120, temperature=0.5)
        if vanilla_restart.startswith(restart_prompt):
            vanilla_restart = vanilla_restart[len(restart_prompt):]

        print(f"\n[VANILLA — restarted with correction]\n{vanilla_restart.strip()}")

        # Duplex: mid-stream correction
        duplex_out, at_correction = duplex_model.generate_with_update(
            prompt_text=prompt,
            max_new_tokens=120,
            temperature=0.5,
            correction_text=correction,
            correction_after_tokens=inject,
        )
        if duplex_out.startswith(SYSTEM):
            duplex_out = duplex_out[len(SYSTEM):]

        print(f"\n[DUPLEX — correction injected at token {inject}]\n{duplex_out.strip()}")

        # Duplex: NO correction (ablation)
        duplex_no_update, _ = duplex_model.generate_with_update(
            prompt_text=prompt,
            max_new_tokens=120,
            temperature=0.5,
        )
        if duplex_no_update.startswith(SYSTEM):
            duplex_no_update = duplex_no_update[len(SYSTEM):]

        print(f"\n[DUPLEX — no correction (ablation)]\n{duplex_no_update.strip()}")
        print("\n")


if __name__ == "__main__":
    main()
