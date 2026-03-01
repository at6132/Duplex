"""
Deep diagnostic: trace the full computation to find where the correction signal is lost.
Measures: encoder outputs, workspace gate values, adapter magnitudes, hidden state deltas.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from duplex.config import DuplexConfig
from duplex.duplex_model import DuplexModel

CKPT = "checkpoints/duplex-1-1.7b/phase2_best.pt"
QWEN = "models/qwen3-1.7b-base"


def main():
    print("Loading model...")
    config = DuplexConfig(qwen_model_path=QWEN, quantize_4bit=False)
    model = DuplexModel(config)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.workspace.load_state_dict(ckpt["workspace_state_dict"])
    model.adapters.load_state_dict(ckpt["adapters_state_dict"])
    model.eval()
    device = next(model.qwen.parameters()).device
    tok = model.tokenizer
    print("Loaded.\n")

    prompt = "Given x = 5, compute 3*x + 7 step by step."
    correction = "Actually, x = 12 not 5. Please recalculate."

    prompt_enc = tok(prompt, return_tensors="pt").to(device)
    correction_enc = tok(correction, return_tensors="pt").to(device)

    with torch.no_grad():
        # =====================================================================
        # 1. ENCODER OUTPUTS
        # =====================================================================
        print("=" * 70)
        print("1. ENCODER OUTPUT ANALYSIS")
        print("=" * 70)

        enc_prompt = model.encoder(prompt_enc["input_ids"], attention_mask=prompt_enc["attention_mask"])
        enc_correction = model.encoder(correction_enc["input_ids"], attention_mask=correction_enc["attention_mask"])

        print(f"  Prompt encoder output:     shape={enc_prompt.shape}, "
              f"mean={enc_prompt.mean().item():.6f}, std={enc_prompt.std().item():.6f}, "
              f"norm={enc_prompt.norm().item():.4f}")
        print(f"  Correction encoder output: shape={enc_correction.shape}, "
              f"mean={enc_correction.mean().item():.6f}, std={enc_correction.std().item():.6f}, "
              f"norm={enc_correction.norm().item():.4f}")

        # =====================================================================
        # 2. WORKSPACE: before and after correction
        # =====================================================================
        print(f"\n{'=' * 70}")
        print("2. WORKSPACE ANALYSIS")
        print("=" * 70)

        ws_prompt = model.workspace(enc_prompt, encoder_mask=prompt_enc["attention_mask"])
        ws_corrected = model.workspace(enc_correction, encoder_mask=correction_enc["attention_mask"], workspace=ws_prompt)

        ws_diff = ws_corrected - ws_prompt
        print(f"  Workspace (prompt only):  mean={ws_prompt.mean().item():.6f}, "
              f"std={ws_prompt.std().item():.6f}, norm={ws_prompt.norm().item():.4f}")
        print(f"  Workspace (+ correction): mean={ws_corrected.mean().item():.6f}, "
              f"std={ws_corrected.std().item():.6f}, norm={ws_corrected.norm().item():.4f}")
        print(f"  DELTA (correction effect): mean={ws_diff.mean().item():.6f}, "
              f"std={ws_diff.std().item():.6f}, norm={ws_diff.norm().item():.4f}")
        print(f"  Delta as % of workspace:   {100 * ws_diff.norm().item() / ws_prompt.norm().item():.2f}%")

        # Gate analysis: re-run workspace.update manually to get gate values
        ws_module = model.workspace
        B = ws_prompt.size(0)
        N = ws_module.n_slots
        w = ws_module.ln_w(ws_prompt)
        e = ws_module.ln_e(enc_correction)
        T_enc = enc_correction.size(1)

        q = ws_module.q_proj(w).view(B, N, ws_module.n_heads, ws_module.head_dim).transpose(1, 2)
        k = ws_module.k_proj(e).view(B, T_enc, ws_module.n_heads, ws_module.head_dim).transpose(1, 2)
        v = ws_module.v_proj(e).view(B, T_enc, ws_module.n_heads, ws_module.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (ws_module.head_dim ** 0.5)
        if correction_enc["attention_mask"] is not None:
            mask = correction_enc["attention_mask"].unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1).nan_to_num(0.0)
        attended = (attn @ v).transpose(1, 2).contiguous().view(B, N, ws_module.inner_dim)
        attended = ws_module.attn_out(attended)

        delta = ws_module.delta_mlp(attended)
        gate = torch.sigmoid(ws_module.gate_linear(torch.cat([ws_prompt, attended], dim=-1)))

        print(f"\n  Gate values:")
        print(f"    mean={gate.mean().item():.6f}, min={gate.min().item():.6f}, "
              f"max={gate.max().item():.6f}, std={gate.std().item():.6f}")
        print(f"    % of gate > 0.5: {100 * (gate > 0.5).float().mean().item():.1f}%")
        print(f"    % of gate > 0.1: {100 * (gate > 0.1).float().mean().item():.1f}%")
        print(f"  Delta (from MLP):")
        print(f"    norm={delta.norm().item():.4f}, mean={delta.mean().item():.6f}")
        print(f"  gate * delta (actual update):")
        actual_update = gate * delta
        print(f"    norm={actual_update.norm().item():.4f}, mean={actual_update.mean().item():.6f}")

        # =====================================================================
        # 3. ADAPTER OUTPUT MAGNITUDES
        # =====================================================================
        print(f"\n{'=' * 70}")
        print("3. ADAPTER OUTPUT ANALYSIS (per layer)")
        print("=" * 70)

        # Get a hidden state by running qwen embedding + first few tokens
        test_input = tok(prompt, return_tensors="pt").to(device)
        embed_out = model.qwen.model.embed_tokens(test_input["input_ids"])

        print(f"  Qwen embedding output norm: {embed_out.norm().item():.4f}")
        print()

        for i, adapter in enumerate(model.adapters):
            # Run adapter with prompt-only workspace
            adapter_out_prompt = adapter(embed_out, ws_prompt)
            # Run adapter with corrected workspace
            adapter_out_corrected = adapter(embed_out, ws_corrected)
            adapter_diff = adapter_out_corrected - adapter_out_prompt

            if i < 5 or i >= 25:  # Show first 5 and last 3 layers
                print(f"  Layer {i:2d}: "
                      f"adapter_norm={adapter_out_corrected.norm().item():.6f}, "
                      f"embed_norm_ratio={adapter_out_corrected.norm().item() / embed_out.norm().item() * 100:.4f}%, "
                      f"correction_delta_norm={adapter_diff.norm().item():.6f}")
            elif i == 5:
                print(f"  ... (layers 5-24 omitted) ...")

        # =====================================================================
        # 4. O_PROJ WEIGHT ANALYSIS (are adapters still near-zero?)
        # =====================================================================
        print(f"\n{'=' * 70}")
        print("4. ADAPTER O_PROJ WEIGHTS (did they train away from zero?)")
        print("=" * 70)

        for i, adapter in enumerate(model.adapters):
            w = adapter.o_proj.weight
            if i < 5 or i >= 25:
                print(f"  Layer {i:2d}: o_proj weight norm={w.norm().item():.6f}, "
                      f"max={w.abs().max().item():.6f}, mean={w.abs().mean().item():.8f}")
            elif i == 5:
                print(f"  ... (layers 5-24 omitted) ...")

        # =====================================================================
        # 5. GENERATION COMPARISON (greedy, deterministic)
        # =====================================================================
        print(f"\n{'=' * 70}")
        print("5. GENERATION TEST (greedy, temp=0.01)")
        print("=" * 70)

        # With correction
        model._current_workspace = ws_corrected
        gen_ids = prompt_enc["input_ids"].clone()
        for step in range(40):
            out = model.qwen(input_ids=gen_ids)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids = torch.cat([gen_ids, next_token], dim=1)
            if next_token.item() == tok.eos_token_id:
                break
        model._current_workspace = None
        text_corrected = tok.decode(gen_ids[0], skip_special_tokens=True)
        print(f"\n  [WITH correction workspace]:")
        print(f"  {text_corrected}")

        # Without correction
        model._current_workspace = ws_prompt
        gen_ids = prompt_enc["input_ids"].clone()
        for step in range(40):
            out = model.qwen(input_ids=gen_ids)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids = torch.cat([gen_ids, next_token], dim=1)
            if next_token.item() == tok.eos_token_id:
                break
        model._current_workspace = None
        text_no_correction = tok.decode(gen_ids[0], skip_special_tokens=True)
        print(f"\n  [WITHOUT correction (prompt only)]:")
        print(f"  {text_no_correction}")

        # No workspace at all
        model._current_workspace = None
        gen_ids = prompt_enc["input_ids"].clone()
        for step in range(40):
            out = model.qwen(input_ids=gen_ids)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids = torch.cat([gen_ids, next_token], dim=1)
            if next_token.item() == tok.eos_token_id:
                break
        text_no_ws = tok.decode(gen_ids[0], skip_special_tokens=True)
        print(f"\n  [NO workspace (vanilla Qwen behavior)]:")
        print(f"  {text_no_ws}")

        # =====================================================================
        # 6. LOGIT COMPARISON at critical token position
        # =====================================================================
        print(f"\n{'=' * 70}")
        print("6. LOGIT ANALYSIS — how much does correction shift predictions?")
        print("=" * 70)

        # Get logits at the 15th generated token with and without correction
        model._current_workspace = ws_corrected
        out_corr = model.qwen(input_ids=prompt_enc["input_ids"])
        logits_corr = out_corr.logits[0, -1, :]  # logits for next token

        model._current_workspace = ws_prompt
        out_no = model.qwen(input_ids=prompt_enc["input_ids"])
        logits_no = out_no.logits[0, -1, :]

        model._current_workspace = None
        out_vanilla = model.qwen(input_ids=prompt_enc["input_ids"])
        logits_vanilla = out_vanilla.logits[0, -1, :]

        logit_diff_corr = (logits_corr - logits_vanilla)
        logit_diff_no = (logits_no - logits_vanilla)

        print(f"  Logit shift (correction vs vanilla): "
              f"mean={logit_diff_corr.abs().mean().item():.6f}, "
              f"max={logit_diff_corr.abs().max().item():.4f}")
        print(f"  Logit shift (prompt-ws vs vanilla):  "
              f"mean={logit_diff_no.abs().mean().item():.6f}, "
              f"max={logit_diff_no.abs().max().item():.4f}")
        print(f"  Logit shift (correction vs prompt-ws): "
              f"mean={(logits_corr - logits_no).abs().mean().item():.6f}, "
              f"max={(logits_corr - logits_no).abs().max().item():.4f}")

        # Top-5 tokens with vs without correction
        probs_corr = F.softmax(logits_corr, dim=-1)
        probs_no = F.softmax(logits_no, dim=-1)
        probs_vanilla = F.softmax(logits_vanilla, dim=-1)

        print(f"\n  Top-5 next token predictions:")
        print(f"  {'Token':<15} {'Vanilla':>10} {'Prompt-WS':>10} {'+Correction':>12}")
        top_tokens = logits_vanilla.topk(10).indices
        for tid in top_tokens:
            t = tok.decode([tid.item()])
            p_v = probs_vanilla[tid].item()
            p_n = probs_no[tid].item()
            p_c = probs_corr[tid].item()
            print(f"  {repr(t):<15} {p_v:>10.4f} {p_n:>10.4f} {p_c:>12.4f}")

        model._current_workspace = None


if __name__ == "__main__":
    main()
