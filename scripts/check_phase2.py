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

CHECKS = [
    {
        "name": "Name correction",
        "prompt": "Write a short bio for James, a 28-year-old chef from Paris.",
        "correction": "Update: the person's name is Marco, not James.",
        "inject_after": 15,
        "must_contain": ["Marco"],
        "must_not_contain": ["James"],
    },
    {
        "name": "City correction",
        "prompt": "Describe a tourist visiting New York for the first time.",
        "correction": "Change the city to Tokyo.",
        "inject_after": 12,
        "must_contain": ["Tokyo"],
        "must_not_contain": [],
    },
    {
        "name": "Profession correction",
        "prompt": "Tell me about Sarah, who works as a lawyer in London.",
        "correction": "Correction: Sarah is actually a nurse, not a lawyer.",
        "inject_after": 14,
        "must_contain": ["nurse"],
        "must_not_contain": [],
    },
    {
        "name": "Number correction",
        "prompt": "A company has 50 employees. Describe its team structure.",
        "correction": "The company actually has 200 employees, not 50.",
        "inject_after": 12,
        "must_contain": ["200"],
        "must_not_contain": [],
    },
]


def run():
    if not os.path.exists(CKPT):
        alt = CKPT.replace("phase2_best", "final")
        if os.path.exists(alt):
            ckpt = alt
        else:
            print(f"ERROR: No checkpoint found at {CKPT}")
            sys.exit(1)
    else:
        ckpt = CKPT

    print(f"Loading Duplex from {ckpt} ...")
    model = load_duplex_model(QWEN, ckpt)
    model.eval()
    print("Loaded.\n")

    print(f"Architecture: deep prefix conditioning / P-Tuning v2")
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
