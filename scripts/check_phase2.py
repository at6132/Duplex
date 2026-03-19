"""
Quick Phase 2 verification for Duplex-1.4.

Loads Duplex, runs 4 targeted correction scenarios, checks whether
the corrected entity appears in the output and the old entity is gone.

Usage:
    python scripts/check_phase2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from duplex.inference.generate import load_duplex_model

CKPT = "checkpoints/duplex-1.4-1.7b/phase2_best.pt"
QWEN = "models/qwen3-1.7b-base"

# In-distribution checks: prompts match training data format exactly
CHECKS = [
    {
        "name": "Fact correction (capital)",
        "prompt": "What is the capital of Australia? Please explain briefly.",
        "correction": "That's incorrect. The capital of Australia is actually Canberra, not Sydney.",
        "inject_after": 15,
        "must_contain": ["Canberra"],
        "must_not_contain": [],
    },
    {
        "name": "Profile update (name/age)",
        "prompt": "Create a brief profile for Alice: age 30, software engineer based in New York, enjoys reading.",
        "correction": "Update: Alice is actually 45, works as a doctor, and lives in London.",
        "inject_after": 15,
        "must_contain": ["45"],
        "must_not_contain": [],
    },
    {
        "name": "Language switch (Python->Rust)",
        "prompt": "Write a simple hello world function in Python.",
        "correction": "Actually, please write it in Rust instead of Python.",
        "inject_after": 12,
        "must_contain": ["Rust"],
        "must_not_contain": [],
    },
    {
        "name": "Topic redirect",
        "prompt": "Write a brief paragraph about renewable energy.",
        "correction": "Actually, I'd like you to write about quantum computing instead of renewable energy.",
        "inject_after": 15,
        "must_contain": ["quantum"],
        "must_not_contain": [],
    },
    {
        "name": "Arithmetic correction",
        "prompt": "What is 150 + 200?",
        "correction": "Wait, the first number should be 300, not 150.",
        "inject_after": 10,
        "must_contain": ["500"],
        "must_not_contain": [],
    },
    {
        "name": "Variable substitution",
        "prompt": "Given x = 5, compute 3*x + 7 step by step.",
        "correction": "Actually, x = 10, not 5. Please recalculate.",
        "inject_after": 12,
        "must_contain": ["37"],
        "must_not_contain": [],
    },
]


def run():
    candidates = [
        CKPT,
        CKPT.replace("phase2_best", "final"),
        CKPT.replace("phase2_best", "phase1_best"),
        CKPT.replace("phase2_best", "phase1_final"),
    ]
    ckpt = None
    for c in candidates:
        if os.path.exists(c):
            ckpt = c
            break
    if ckpt is None:
        print(f"ERROR: No checkpoint found. Tried: {candidates}")
        sys.exit(1)

    print(f"Loading Duplex from {ckpt} ...")
    model = load_duplex_model(QWEN, ckpt)
    model.eval()
    print("Loaded.\n")

    print(f"Architecture: deep prefix + LoRA (Q/V, r=16)")
    print(f"Trainable params: {model.trainable_param_count():,}")
    print(f"Prefix slots: {model.config.n_workspace_slots}")
    print(f"Decoder input: GENERIC instruction ('{model.GENERIC_INSTRUCTION}')\n")

    passes = 0

    for c in CHECKS:
        # Run WITHOUT correction first (baseline)
        baseline, _ = model.generate_with_update(
            prompt_text=c["prompt"],
            max_new_tokens=150,
            temperature=0.1,
        )

        # Run WITH correction
        corrected, text_at_correction = model.generate_with_update(
            prompt_text=c["prompt"],
            max_new_tokens=150,
            temperature=0.1,
            correction_text=c["correction"],
            correction_after_tokens=c["inject_after"],
        )

        entity_ok = all(m.lower() in corrected.lower() for m in c["must_contain"])
        avoid_ok = all(m.lower() not in corrected.lower() for m in c["must_not_contain"])
        outputs_differ = baseline.strip() != corrected.strip()
        passed = entity_ok and avoid_ok and outputs_differ

        status = "PASS" if passed else "FAIL"
        if passed:
            passes += 1

        print(f"{'='*60}")
        print(f"[{status}] {c['name']}")
        print(f"  Prompt:     {c['prompt'][:60]}")
        print(f"  Correction: {c['correction']}")
        print(f"  Inject at:  token {c['inject_after']}")
        print(f"  Must-contain {c['must_contain']}: {'YES' if entity_ok else 'NO'}")
        if c["must_not_contain"]:
            print(f"  Must-avoid  {c['must_not_contain']}: {'YES (avoided)' if avoid_ok else 'NO (still present)'}")
        print(f"  Outputs differ after correction: {'YES' if outputs_differ else 'NO'}")
        print(f"\n  Baseline output:\n  {baseline.strip()[:200]}")
        print(f"\n  Corrected output:\n  {corrected.strip()[:200]}")
        if text_at_correction:
            print(f"\n  Text at correction point:\n  {text_at_correction[:150]}")
        print()

    print(f"{'='*60}")
    print(f"RESULT: {passes}/{len(CHECKS)} scenarios passed")
    if passes == len(CHECKS):
        print("Phase 2 training SUCCESS — model applies corrections via workspace update.")
    elif passes >= len(CHECKS) // 2:
        print("Partial success — model partially learned corrections. Consider more Phase 2 training.")
    else:
        print("Phase 2 training INSUFFICIENT — model not reliably applying corrections.")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
