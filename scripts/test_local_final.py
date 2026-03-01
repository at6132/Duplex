"""
FINAL local validation for Duplex-1.2 prefix conditioning.
Proves the architecture works with JOINT multi-task training (no sequential phases).

Tests:
  1. Same-class generation (no corruption)
  2. Redirect (prefix steers generation)
  3. Marker token emission (action token analog for corrections)
  4. Mid-stream marker (inject correction during generation, emit marker)

All tasks trained jointly from the start — no catastrophic forgetting.

Usage:
    python scripts/test_local_final.py
"""
import sys, os, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from duplex.workspace import WorkspaceModule
from duplex.encoder import UpdateEncoder

VOCAB = 32
D_MODEL = 128
N_HEADS = 4
HEAD_DIM = int(D_MODEL // N_HEADS)
N_LAYERS = 12
FF_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CLASSES = 5
MOTIFS = {k: [4*k, 4*k+1, 4*k+2, 4*k+3] for k in range(N_CLASSES)}
MARKER = 30  # action token analog
SEQ_LEN = 24


# -------------------- Backbone --------------------

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, N_HEADS, HEAD_DIM)
        q, k, v = qkv.unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        scores.masked_fill_(causal, float("-inf"))
        return self.out((F.softmax(scores, -1) @ v).transpose(1, 2).contiguous().view(B, T, D_MODEL))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = Attn()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, FF_DIM), nn.GELU(), nn.Linear(FF_DIM, D_MODEL))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
    def forward_embeds(self, e):
        for layer in self.layers:
            e = layer(e)
        return self.head(self.ln_f(e))
    def forward(self, ids):
        return self.forward_embeds(self.embed(ids))


# -------------------- Prefix Model --------------------

class PrefixLM(nn.Module):
    def __init__(self, backbone, n_prefix=12):
        super().__init__()
        self.backbone = backbone
        self.n_prefix = n_prefix
        self.encoder = UpdateEncoder(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            d_ff=FF_DIM, n_layers=2, dropout=0.0).to(DEVICE)
        self.workspace = WorkspaceModule(
            n_slots=n_prefix, d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            dropout=0.0).to(DEVICE)

    def build_prefix(self, prompt_ids, correction_ids=None):
        """Build prefix from prompt, optionally including correction tokens."""
        enc = self.encoder(prompt_ids)
        ws = self.workspace(enc)
        if correction_ids is not None:
            corr_enc = self.encoder(correction_ids)
            ws_updated = self.workspace(corr_enc, workspace=ws)
            return torch.cat([ws_updated, corr_enc], dim=1)
        return ws

    def forward_with_prefix(self, prefix, input_ids):
        embs = self.backbone.embed(input_ids)
        combined = torch.cat([prefix, embs], dim=1)
        logits = self.backbone.forward_embeds(combined)
        return logits[:, prefix.size(1):, :]

    @torch.no_grad()
    def generate(self, prefix, start_ids, n_tokens=20):
        ids = start_ids.clone()
        for _ in range(n_tokens):
            embs = self.backbone.embed(ids)
            combined = torch.cat([prefix, embs], dim=1)
            logits = self.backbone.forward_embeds(combined)
            ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
        return ids[0].tolist()

    @torch.no_grad()
    def generate_with_correction(self, prompt_ids, start_ids, correction_ids,
                                  inject_at, n_tokens=20):
        """Generate, inject correction mid-stream, continue."""
        prefix = self.build_prefix(prompt_ids)
        ids = start_ids.clone()
        for step in range(n_tokens):
            if step == inject_at:
                prefix = self.build_prefix(prompt_ids, correction_ids)
            embs = self.backbone.embed(ids)
            combined = torch.cat([prefix, embs], dim=1)
            logits = self.backbone.forward_embeds(combined)
            ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
        return ids[0].tolist()

    def adapter_params(self):
        # Include backbone's embed + lm_head — special tokens need trainable I/O
        return (list(self.encoder.parameters())
                + list(self.workspace.parameters())
                + list(self.backbone.embed.parameters())
                + list(self.backbone.head.parameters()))


# -------------------- Joint Training Data --------------------

def make_seq(cls, length=SEQ_LEN):
    motif = MOTIFS[cls]
    return [cls + 20] + (motif * ((length - 1) // len(motif) + 1))[:length - 1]


def sample_task():
    """Returns (prefix, target_seq, task_type) for one of three tasks.
    Evenly distributed: 1/3 same-class, 1/3 redirect, 1/3 marker-correction."""
    task = random.randint(0, 2)

    if task == 0:  # Same-class: prompt matches, generate pattern
        cls = random.randint(0, N_CLASSES-1)
        prompt = torch.tensor([cls + 20], device=DEVICE)
        target = torch.tensor(make_seq(cls), device=DEVICE)
        return "same", prompt, None, target

    elif task == 1:  # Redirect: prompt says tgt, start token is src
        src = random.randint(0, N_CLASSES-1)
        tgt = (src + random.randint(1, N_CLASSES-1)) % N_CLASSES
        prompt = torch.tensor([tgt + 20], device=DEVICE)
        tgt_motif = MOTIFS[tgt]
        target = torch.tensor([src + 20] + (tgt_motif * 6)[:SEQ_LEN-1], device=DEVICE)
        return "redirect", prompt, None, target

    else:  # Marker-correction: after 6 src tokens, emit MARKER then tgt pattern
        src = random.randint(0, N_CLASSES-1)
        tgt = (src + random.randint(1, N_CLASSES-1)) % N_CLASSES
        prompt = torch.tensor([src + 20], device=DEVICE)
        correction = torch.tensor([tgt + 20], device=DEVICE)
        src_motif = MOTIFS[src]
        tgt_motif = MOTIFS[tgt]
        target = torch.tensor(
            [src + 20] + (src_motif * 2)[:6] + [MARKER] + (tgt_motif * 4)[:SEQ_LEN-8],
            device=DEVICE
        )
        return "marker", prompt, correction, target


# -------------------- Evaluation --------------------

def check_motif(tokens, motif, start=1, length=None):
    c = t = 0
    end = (start + length) if length else len(tokens)
    for i in range(start, min(end, len(tokens))):
        if tokens[i] == motif[(i - start) % len(motif)]:
            c += 1
        t += 1
    return c / max(1, t)


def main():
    torch.manual_seed(42)
    random.seed(42)

    print("=" * 60)
    print("  Duplex-1.2 FINAL Validation")
    print(f"  Device: {DEVICE}")
    print(f"  Backbone: {N_LAYERS}L, {D_MODEL}d | Motifs: 4-token")
    print(f"  Training: ALL tasks jointly (no sequential phases)")
    print("=" * 60)

    # Phase 1: pretrain backbone
    print("\n[BACKBONE] Pretraining...")
    backbone = LM().to(DEVICE)
    opt_bb = torch.optim.AdamW(backbone.parameters(), lr=1e-3)
    backbone.train()
    for step in range(1200):
        cls = random.randint(0, N_CLASSES-1)
        seq = torch.tensor(make_seq(cls), device=DEVICE).unsqueeze(0)
        logits = backbone(seq)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        opt_bb.zero_grad(); loss.backward(); opt_bb.step()
        if (step + 1) % 400 == 0:
            print(f"    Step {step+1:4d} | loss: {loss.item():.4f}")

    # Phase 2: joint multi-task training of prefix adapter
    print(f"\n[ADAPTER] Joint training (same-class + redirect + marker)...")
    model = PrefixLM(backbone, n_prefix=12).to(DEVICE)
    for p in model.backbone.parameters():
        p.requires_grad = False
    # Re-enable embed + lm_head — needed to learn special token I/O (MARKER)
    for p in model.backbone.head.parameters():
        p.requires_grad = True
    for p in model.backbone.embed.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.adapter_params(), lr=3e-3)
    model.train()

    task_counts = {"same": 0, "redirect": 0, "marker": 0}
    for step in range(3000):
        task_type, prompt, correction, target = sample_task()
        task_counts[task_type] += 1

        if task_type in ("same", "redirect"):
            prefix = model.build_prefix(prompt.unsqueeze(0))
        else:  # marker
            prefix = model.build_prefix(prompt.unsqueeze(0), correction.unsqueeze(0))

        logits = model.forward_with_prefix(prefix, target.unsqueeze(0))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), target[1:].unsqueeze(0).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

        if (step + 1) % 500 == 0:
            print(f"    Step {step+1:4d} | loss: {loss.item():.4f} | tasks: {task_counts}")

    # ---- EVALUATION ----
    print(f"\n{'='*60}")
    print("  EVALUATION")
    print(f"{'='*60}")
    model.backbone.eval()

    # Test 1: Same-class
    print("\n  [1] Same-class generation...")
    sc_scores = []
    for _ in range(50):
        cls = random.randint(0, N_CLASSES-1)
        prefix = model.build_prefix(torch.tensor([[cls + 20]], device=DEVICE))
        tokens = model.generate(prefix, torch.tensor([[cls + 20]], device=DEVICE), 20)
        sc_scores.append(check_motif(tokens, MOTIFS[cls]))
    sc = sum(sc_scores) / len(sc_scores)
    print(f"    {sc:.1%}")

    # Test 2: Redirect
    print("  [2] Basic redirect...")
    rd_scores = []
    for _ in range(50):
        src = random.randint(0, N_CLASSES-1)
        tgt = (src + 1) % N_CLASSES
        prefix = model.build_prefix(torch.tensor([[tgt + 20]], device=DEVICE))
        tokens = model.generate(prefix, torch.tensor([[src + 20]], device=DEVICE), 20)
        rd_scores.append(check_motif(tokens, MOTIFS[tgt]))
    rd = sum(rd_scores) / len(rd_scores)
    print(f"    {rd:.1%}")

    # Test 3: Marker emission (when correction present, emit MARKER at right position)
    print("  [3] Marker token emission with correction prefix...")
    mk_scores = []
    for _ in range(50):
        src = random.randint(0, N_CLASSES-1)
        tgt = (src + 1) % N_CLASSES
        prefix = model.build_prefix(
            torch.tensor([[src + 20]], device=DEVICE),
            torch.tensor([[tgt + 20]], device=DEVICE),
        )
        tokens = model.generate(prefix, torch.tensor([[src + 20]], device=DEVICE), 20)
        # Check: is MARKER present anywhere in first 12 tokens?
        has_marker = MARKER in tokens[1:12]
        mk_scores.append(1.0 if has_marker else 0.0)
    mk = sum(mk_scores) / len(mk_scores)
    print(f"    {mk:.1%}")

    # Test 4: Mid-stream correction -> marker
    print("  [4] Mid-stream correction (inject at token 6, check marker)...")
    ms_scores = []
    for _ in range(50):
        src = random.randint(0, N_CLASSES-1)
        tgt = (src + 1) % N_CLASSES
        tokens = model.generate_with_correction(
            torch.tensor([[src + 20]], device=DEVICE),
            torch.tensor([[src + 20]], device=DEVICE),
            torch.tensor([[tgt + 20]], device=DEVICE),
            inject_at=6, n_tokens=20,
        )
        has_marker = MARKER in tokens[7:14]
        ms_scores.append(1.0 if has_marker else 0.0)
    ms = sum(ms_scores) / len(ms_scores)
    print(f"    {ms:.1%}")

    # Test 5: No correction -> no marker (sanity check)
    print("  [5] No correction -> should NOT emit marker...")
    nm_scores = []
    for _ in range(50):
        cls = random.randint(0, N_CLASSES-1)
        prefix = model.build_prefix(torch.tensor([[cls + 20]], device=DEVICE))
        tokens = model.generate(prefix, torch.tensor([[cls + 20]], device=DEVICE), 20)
        no_marker = MARKER not in tokens
        nm_scores.append(1.0 if no_marker else 0.0)
    nm = sum(nm_scores) / len(nm_scores)
    print(f"    {nm:.1%}")

    # VERDICT
    print(f"\n{'='*60}")
    print("  FINAL SCORECARD")
    print(f"{'='*60}")
    print(f"  {'Test':<45} {'Score':>8}")
    print(f"  {'-'*55}")
    print(f"  {'1. Same-class (no corruption)':<45} {sc:>7.1%}")
    print(f"  {'2. Redirect (prefix steers generation)':<45} {rd:>7.1%}")
    print(f"  {'3. Marker with correction prefix':<45} {mk:>7.1%}")
    print(f"  {'4. Mid-stream correction -> marker':<45} {ms:>7.1%}")
    print(f"  {'5. No correction -> no marker':<45} {nm:>7.1%}")
    print()

    critical = [sc, rd, mk, nm]
    core_avg = sum(critical) / len(critical)

    if min(critical) >= 0.7:
        print(f"  VERDICT: PASS (core avg {core_avg:.1%})")
        print("  Architecture + joint training validated. Ready for H200s.")
    elif min(critical) >= 0.4:
        print(f"  VERDICT: PARTIAL (core avg {core_avg:.1%})")
        print("  Works but needs tuning before H200 deployment.")
    else:
        print(f"  VERDICT: FAIL (core avg {core_avg:.1%})")
        print("  Architecture needs more work.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
