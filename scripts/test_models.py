"""Quick CLI test for Duplex v2: run scenarios and check if corrections work."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from duplex.inference.generate import load_vanilla_qwen, load_duplex_model, generate_vanilla
from duplex.renderer import render_final, strip_action_tokens

SYSTEM = (
    "You are a helpful assistant. Answer the user's question directly and concisely. "
    "When you receive new information or a correction mid-response, briefly acknowledge it "
    "and then continue with the updated information. Do not generate any extra conversation turns.\n\n"
)

SCENARIOS = [
    {
        "name": "Fact correction",
        "prompt": "What is the capital of Australia? Explain briefly.",
        "correction": "That's wrong -- the capital is Canberra, not Sydney.",
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
    ckpt_path = "checkpoints/duplex-1.2-1.7b/phase2_best.pt"

    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/duplex-1.2-1.7b/final.pt"
    if not os.path.exists(qwen_path):
        print(f"ERROR: {qwen_path} not found.")
        return
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found.")
        return

    print("Loading vanilla Qwen...")
    qwen_model, qwen_tok = load_vanilla_qwen(qwen_path, quantize=False)

    print("Loading Duplex v2...")
    duplex_model = load_duplex_model(qwen_path, ckpt_path)
    print("Both loaded.\n")

    print(f"Architecture: prefix conditioning")
    print(f"Trainable params: {duplex_model.trainable_param_count():,}")
    print()

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

        # Vanilla (no correction)
        vanilla_out = generate_vanilla(qwen_model, qwen_tok, prompt, max_new_tokens=120, temperature=0.3)
        if vanilla_out.startswith(SYSTEM):
            vanilla_out = vanilla_out[len(SYSTEM):]
        print(f"\n[VANILLA - no correction]\n{vanilla_out.strip()}")

        # Duplex with correction
        duplex_raw, at_corr = duplex_model.generate_with_update(
            prompt_text=prompt,
            max_new_tokens=120,
            temperature=0.3,
            correction_text=correction,
            correction_after_tokens=inject,
        )
        if duplex_raw.startswith(SYSTEM):
            duplex_raw = duplex_raw[len(SYSTEM):]
        duplex_rendered = render_final(duplex_raw)
        print(f"\n[DUPLEX - with correction (raw)]\n{strip_action_tokens(duplex_raw).strip()}")
        print(f"\n[DUPLEX - with correction (rendered)]\n{duplex_rendered.strip()}")

        # Duplex without correction (ablation)
        duplex_no, _ = duplex_model.generate_with_update(
            prompt_text=prompt,
            max_new_tokens=120,
            temperature=0.3,
        )
        if duplex_no.startswith(SYSTEM):
            duplex_no = duplex_no[len(SYSTEM):]
        print(f"\n[DUPLEX - no correction (ablation)]\n{strip_action_tokens(duplex_no).strip()}")
        print("\n")


if __name__ == "__main__":
    main()
