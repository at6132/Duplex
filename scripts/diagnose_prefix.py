"""
Diagnostic: pinpoint WHERE prompt information is lost in the Duplex pipeline.

Checks:
  1. Encoder outputs: are they different for different prompts?
  2. Workspace outputs: do they collapse to similar vectors?
  3. Deep prefix K/V: do different prompts produce different K/V?
  4. First-token logits: does the prefix change what Qwen predicts?
  5. Cross-prompt comparison: do different prompts produce different outputs?

Usage (on pod):
    python scripts/diagnose_prefix.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from duplex.inference.generate import load_duplex_model

CKPT = "checkpoints/duplex-1.4-1.7b/phase2_best.pt"
ALT_CKPT = "checkpoints/duplex-1.4-1.7b/final.pt"
QWEN = "models/qwen3-1.7b-base"

PROMPTS = [
    "Write a short bio for James, a 28-year-old chef from Paris.",
    "What is the capital of Australia? Please explain briefly.",
    "Write a simple hello world function in Python.",
    "A company has 50 employees. Describe its team structure.",
]


def cosine_sim(a, b):
    """Cosine similarity between two tensors, mean-pooled if different sizes."""
    if a.shape != b.shape:
        a_vec = a.float().mean(dim=1).reshape(-1)
        b_vec = b.float().mean(dim=1).reshape(-1)
    else:
        a_vec = a.reshape(-1).float()
        b_vec = b.reshape(-1).float()
    return F.cosine_similarity(a_vec.unsqueeze(0), b_vec.unsqueeze(0)).item()


def main():
    ckpt = CKPT if os.path.exists(CKPT) else ALT_CKPT
    print(f"Loading from {ckpt} ...")
    model = load_duplex_model(QWEN, ckpt)
    model.eval()
    device = next(model.qwen.parameters()).device

    print(f"\nTrainable params: {model.trainable_param_count():,}")
    print(f"Prefix slots: {model.config.n_workspace_slots}")
    print(f"Deep prefix layers: {model.config.n_decoder_layers}")

    # ---- Step 1: Encode prompts ----
    print(f"\n{'='*70}")
    print("  STEP 1: Encoder outputs — are they different per prompt?")
    print(f"{'='*70}")

    encoder_outputs = []
    workspace_outputs = []
    prefix_caches = []

    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            enc = model.tokenizer(prompt, return_tensors="pt").to(device)
            encoder_out = model.encoder(enc["input_ids"], attention_mask=enc["attention_mask"])
            ws = model.workspace(encoder_out, encoder_mask=enc["attention_mask"])
            cache = model._build_deep_cache(ws)

            encoder_outputs.append(encoder_out)
            workspace_outputs.append(ws)
            prefix_caches.append(cache)

            enc_norm = encoder_out.float().norm().item()
            ws_norm = ws.float().norm().item()
            print(f"  Prompt {i}: encoder_norm={enc_norm:.2f}, workspace_norm={ws_norm:.2f}")
            print(f"    '{prompt[:50]}...'")

    print(f"\n  Encoder pairwise cosine similarity:")
    for i in range(len(PROMPTS)):
        for j in range(i+1, len(PROMPTS)):
            sim = cosine_sim(encoder_outputs[i], encoder_outputs[j])
            print(f"    Prompt {i} vs {j}: {sim:.4f}")

    # ---- Step 2: Workspace outputs ----
    print(f"\n{'='*70}")
    print("  STEP 2: Workspace outputs — do they collapse?")
    print(f"{'='*70}")

    print(f"  Workspace pairwise cosine similarity:")
    for i in range(len(PROMPTS)):
        for j in range(i+1, len(PROMPTS)):
            sim = cosine_sim(workspace_outputs[i], workspace_outputs[j])
            status = "COLLAPSED" if sim > 0.95 else "DIVERSE" if sim < 0.8 else "MODERATE"
            print(f"    Prompt {i} vs {j}: {sim:.4f}  [{status}]")

    # ---- Step 3: Deep prefix K/V ----
    print(f"\n{'='*70}")
    print("  STEP 3: Deep prefix K/V — different per prompt?")
    print(f"{'='*70}")

    def get_kv(cache, layer_idx):
        if hasattr(cache, 'key_cache'):
            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
        if hasattr(cache, '_key_cache'):
            return cache._key_cache[layer_idx], cache._value_cache[layer_idx]
        # Fallback: try to_legacy_cache
        legacy = cache.to_legacy_cache() if hasattr(cache, 'to_legacy_cache') else None
        if legacy is not None:
            return legacy[layer_idx][0], legacy[layer_idx][1]
        raise RuntimeError(f"Cannot access DynamicCache internals. Attrs: {dir(cache)}")

    for layer_idx in [0, 13, 27]:
        print(f"\n  Layer {layer_idx}:")
        for i in range(len(PROMPTS)):
            for j in range(i+1, len(PROMPTS)):
                ki, vi = get_kv(prefix_caches[i], layer_idx)
                kj, vj = get_kv(prefix_caches[j], layer_idx)
                k_sim = cosine_sim(ki, kj)
                v_sim = cosine_sim(vi, vj)
                print(f"    Prompt {i} vs {j}: K_sim={k_sim:.4f}, V_sim={v_sim:.4f}")

    # ---- Step 4: First-token logits ----
    print(f"\n{'='*70}")
    print("  STEP 4: First-token logits — does prefix change predictions?")
    print(f"{'='*70}")

    generic_text = model.GENERIC_INSTRUCTION
    generic_enc = model.tokenizer(generic_text, return_tensors="pt").to(device)
    generic_ids = generic_enc["input_ids"]
    n_prefix = model.config.n_workspace_slots

    with torch.no_grad():
        # Without prefix
        out_no_prefix = model.qwen(input_ids=generic_ids)
        logits_no_prefix = out_no_prefix.logits[0, -1, :]
        top5_no = torch.topk(logits_no_prefix, 5)
        tokens_no = [model.tokenizer.decode([t]) for t in top5_no.indices]

        print(f"\n  WITHOUT prefix (vanilla Qwen):")
        print(f"    Top-5: {list(zip(tokens_no, [f'{p:.3f}' for p in F.softmax(top5_no.values.float(), dim=-1).tolist()]))}")

        # With prefix for each prompt
        for i, prompt in enumerate(PROMPTS):
            prefix_cache = prefix_caches[i]
            prefix_mask = torch.ones(1, n_prefix, device=device, dtype=torch.long)
            text_mask = torch.ones(1, generic_ids.size(1), device=device, dtype=torch.long)
            attn_mask = torch.cat([prefix_mask, text_mask], dim=1)

            out_with = model.qwen(
                input_ids=generic_ids,
                attention_mask=attn_mask,
                past_key_values=prefix_cache,
            )
            logits_with = out_with.logits[0, -1, :]
            top5_with = torch.topk(logits_with, 5)
            tokens_with = [model.tokenizer.decode([t]) for t in top5_with.indices]

            kl_div = F.kl_div(
                F.log_softmax(logits_with.float(), dim=-1),
                F.softmax(logits_no_prefix.float(), dim=-1),
                reduction="sum",
            ).item()

            print(f"\n  WITH prefix (prompt {i}: '{prompt[:40]}...'):")
            print(f"    Top-5: {list(zip(tokens_with, [f'{p:.3f}' for p in F.softmax(top5_with.values.float(), dim=-1).tolist()]))}")
            print(f"    KL divergence from no-prefix: {kl_div:.4f}")

    # ---- Step 5: Cross-prompt comparison ----
    print(f"\n{'='*70}")
    print("  STEP 5: Cross-prompt logit similarity")
    print(f"{'='*70}")

    all_logits = []
    with torch.no_grad():
        for i in range(len(PROMPTS)):
            prefix_cache = prefix_caches[i]
            prefix_mask = torch.ones(1, n_prefix, device=device, dtype=torch.long)
            text_mask = torch.ones(1, generic_ids.size(1), device=device, dtype=torch.long)
            attn_mask = torch.cat([prefix_mask, text_mask], dim=1)

            out = model.qwen(
                input_ids=generic_ids,
                attention_mask=attn_mask,
                past_key_values=prefix_cache,
            )
            all_logits.append(out.logits[0, -1, :])

    print(f"  Pairwise cosine similarity of first-token logit distributions:")
    for i in range(len(PROMPTS)):
        for j in range(i+1, len(PROMPTS)):
            sim = F.cosine_similarity(
                all_logits[i].float().unsqueeze(0),
                all_logits[j].float().unsqueeze(0)
            ).item()
            print(f"    Prompt {i} vs {j}: {sim:.4f}")

    # ---- VERDICT ----
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")

    ws_sims = []
    for i in range(len(PROMPTS)):
        for j in range(i+1, len(PROMPTS)):
            ws_sims.append(cosine_sim(workspace_outputs[i], workspace_outputs[j]))
    avg_ws_sim = sum(ws_sims) / len(ws_sims)

    logit_sims = []
    for i in range(len(PROMPTS)):
        for j in range(i+1, len(PROMPTS)):
            logit_sims.append(F.cosine_similarity(
                all_logits[i].float().unsqueeze(0),
                all_logits[j].float().unsqueeze(0)
            ).item())
    avg_logit_sim = sum(logit_sims) / len(logit_sims)

    print(f"  Avg workspace cosine sim:     {avg_ws_sim:.4f}")
    print(f"  Avg first-token logit sim:    {avg_logit_sim:.4f}")

    if avg_ws_sim > 0.95:
        print(f"\n  >> DIAGNOSIS (A): Workspace COLLAPSED — all prompts produce ~identical workspace.")
        print(f"     Fix: encoder/workspace is the bottleneck.")
    elif avg_logit_sim > 0.99:
        print(f"\n  >> DIAGNOSIS (B): Workspace is diverse but Qwen IGNORES prefix K/V.")
        print(f"     Fix: add LoRA to Qwen's attention (Q/V projections).")
    elif avg_logit_sim > 0.95:
        print(f"\n  >> DIAGNOSIS (C): Prefix has WEAK influence on Qwen's logits.")
        print(f"     Fix: LoRA or stronger deep prefix projections.")
    else:
        print(f"\n  >> DIAGNOSIS (D): Prefix IS influencing logits meaningfully.")
        print(f"     The issue may be in generation dynamics, not the prefix itself.")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
