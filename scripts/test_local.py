"""
Local architecture validation for Duplex-1.2.
Tests three approaches to adapter conditioning:
  A) Cross-attention adapters (residual to hidden states) — current
  B) Prefix conditioning (workspace as soft input tokens) — proposed
  C) Hybrid (prefix + lightweight cross-attn)

Runs on CPU/GPU in ~3 min. No pretrained model needed.

Usage:
    python scripts/test_local.py
"""
import sys, os, math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from duplex.adapter import CrossAttentionAdapter
from duplex.workspace import WorkspaceModule
from duplex.encoder import UpdateEncoder

VOCAB = 20
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = int(D_MODEL // N_HEADS)
N_LAYERS = 6
FF_DIM = 128
SEQ_LEN = 20

PATTERNS = {k: [2*k, 2*k+1] for k in range(5)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- Tiny causal LM --------------------

class TinyAttn(nn.Module):
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


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = TinyAttn()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, FF_DIM), nn.GELU(), nn.Linear(FF_DIM, D_MODEL))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList([TinyBlock() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward_embeds(self, embeds):
        """Forward pass from embeddings (supports prefix injection)."""
        h = embeds
        for layer in self.layers:
            h = layer(h)
        return self.head(self.ln_f(h))

    def forward(self, ids):
        return self.forward_embeds(self.embed(ids))

    @torch.no_grad()
    def generate(self, start_ids, n_tokens=16):
        ids = start_ids.clone()
        for _ in range(n_tokens):
            logits = self.forward(ids)
            ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
        return ids[0].tolist()


# -------------------- Data --------------------

def make_pattern_data(n=500):
    data = []
    for _ in range(n):
        cls = torch.randint(0, 5, (1,)).item()
        motif = PATTERNS[cls]
        seq = [cls + 10] + (motif * ((SEQ_LEN - 1) // 2 + 1))[:SEQ_LEN - 1]
        data.append(torch.tensor(seq, device=DEVICE))
    return data


def make_redirect_data(n=500):
    data = []
    for _ in range(n):
        src_cls = torch.randint(0, 5, (1,)).item()
        tgt_cls = (src_cls + torch.randint(1, 5, (1,)).item()) % 5
        tgt_motif = PATTERNS[tgt_cls]
        prompt = torch.tensor([tgt_cls + 10], device=DEVICE)
        inp_seq = [src_cls + 10] + (tgt_motif * ((SEQ_LEN - 1) // 2 + 1))[:SEQ_LEN - 1]
        data.append((prompt, torch.tensor(inp_seq, device=DEVICE)))
    return data


# -------------------- APPROACH A: Cross-attention adapters --------------------

class CrossAttnModel(nn.Module):
    def __init__(self, backbone: TinyLM, interval: int = 2):
        super().__init__()
        self.backbone = backbone
        self.adapter_layer_indices = list(range(0, N_LAYERS, interval))
        n_adp = len(self.adapter_layer_indices)

        self.encoder = UpdateEncoder(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            d_ff=FF_DIM, n_layers=2, dropout=0.0).to(DEVICE)
        self.workspace = WorkspaceModule(
            n_slots=8, d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            dropout=0.0).to(DEVICE)
        self.adapters = nn.ModuleList([
            CrossAttentionAdapter(d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
                                  dropout=0.0) for _ in range(n_adp)
        ]).to(DEVICE)
        self.gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1, device=DEVICE)) for _ in range(n_adp)
        ])
        self._ws = None
        self._inject()

    def _inject(self):
        for aidx, lidx in enumerate(self.adapter_layer_indices):
            layer = self.backbone.layers[lidx]
            adapter, gate = self.adapters[aidx], self.gates[aidx]
            orig = layer.forward
            def make_p(fn, a, g):
                def f(x):
                    h = fn(x)
                    if self._ws is not None:
                        h = h + torch.tanh(g).to(h.dtype) * a(h, self._ws)
                    return h
                return f
            layer.forward = make_p(orig, adapter, gate)

    def forward(self, ids, prompt):
        enc = self.encoder(prompt)
        self._ws = self.workspace(enc)
        logits = self.backbone(ids)
        self._ws = None
        return logits

    @torch.no_grad()
    def generate(self, prompt, start_ids, n_tokens=16):
        enc = self.encoder(prompt)
        self._ws = self.workspace(enc)
        ids = start_ids.clone()
        for _ in range(n_tokens):
            logits = self.backbone(ids)
            ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
        self._ws = None
        return ids[0].tolist()

    def adapter_params(self):
        p = list(self.encoder.parameters()) + list(self.workspace.parameters())
        p += list(self.adapters.parameters()) + list(self.gates)
        return p

    def gate_str(self):
        v = [torch.tanh(g).item() for g in self.gates]
        return f"min={min(v):.3f} max={max(v):.3f}"


# -------------------- APPROACH B: Prefix conditioning --------------------

class PrefixModel(nn.Module):
    """Workspace slots are prepended as soft tokens to the input.
    The backbone's self-attention naturally conditions on them."""
    def __init__(self, backbone: TinyLM, n_prefix: int = 8):
        super().__init__()
        self.backbone = backbone
        self.n_prefix = n_prefix

        self.encoder = UpdateEncoder(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            d_ff=FF_DIM, n_layers=2, dropout=0.0).to(DEVICE)
        self.workspace = WorkspaceModule(
            n_slots=n_prefix, d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            dropout=0.0).to(DEVICE)

    def forward(self, ids, prompt):
        enc = self.encoder(prompt)
        prefix = self.workspace(enc)  # (B, n_prefix, D)
        token_embs = self.backbone.embed(ids)  # (B, T, D)
        combined = torch.cat([prefix, token_embs], dim=1)  # (B, n_prefix+T, D)
        logits = self.backbone.forward_embeds(combined)
        return logits[:, self.n_prefix:, :]  # only return logits for real tokens

    @torch.no_grad()
    def generate(self, prompt, start_ids, n_tokens=16):
        enc = self.encoder(prompt)
        prefix = self.workspace(enc)
        ids = start_ids.clone()
        for _ in range(n_tokens):
            token_embs = self.backbone.embed(ids)
            combined = torch.cat([prefix, token_embs], dim=1)
            logits = self.backbone.forward_embeds(combined)
            next_logits = logits[:, -1:]
            ids = torch.cat([ids, next_logits.argmax(-1)], 1)
        return ids[0].tolist()

    def adapter_params(self):
        return list(self.encoder.parameters()) + list(self.workspace.parameters())

    def gate_str(self):
        return "N/A (prefix)"


# -------------------- APPROACH C: Prefix + lightweight cross-attn --------------------

class HybridModel(nn.Module):
    """Prefix for main conditioning + one cross-attention adapter at last layer."""
    def __init__(self, backbone: TinyLM, n_prefix: int = 8):
        super().__init__()
        self.backbone = backbone
        self.n_prefix = n_prefix

        self.encoder = UpdateEncoder(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            d_ff=FF_DIM, n_layers=2, dropout=0.0).to(DEVICE)
        self.workspace = WorkspaceModule(
            n_slots=n_prefix, d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            dropout=0.0).to(DEVICE)

        self.xattn = CrossAttentionAdapter(d_model=D_MODEL, n_heads=N_HEADS,
                                           head_dim=HEAD_DIM, dropout=0.0).to(DEVICE)
        self.gate = nn.Parameter(torch.zeros(1, device=DEVICE))
        self._ws = None
        self._inject_last_layer()

    def _inject_last_layer(self):
        layer = self.backbone.layers[-1]
        orig = layer.forward
        adapter, gate = self.xattn, self.gate
        def patched(x):
            h = orig(x)
            if self._ws is not None:
                h = h + torch.tanh(gate).to(h.dtype) * adapter(h, self._ws)
            return h
        layer.forward = patched

    def forward(self, ids, prompt):
        enc = self.encoder(prompt)
        prefix = self.workspace(enc)
        self._ws = prefix
        token_embs = self.backbone.embed(ids)
        combined = torch.cat([prefix, token_embs], dim=1)
        logits = self.backbone.forward_embeds(combined)
        self._ws = None
        return logits[:, self.n_prefix:, :]

    @torch.no_grad()
    def generate(self, prompt, start_ids, n_tokens=16):
        enc = self.encoder(prompt)
        prefix = self.workspace(enc)
        self._ws = prefix
        ids = start_ids.clone()
        for _ in range(n_tokens):
            token_embs = self.backbone.embed(ids)
            combined = torch.cat([prefix, token_embs], dim=1)
            logits = self.backbone.forward_embeds(combined)
            ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
        self._ws = None
        return ids[0].tolist()

    def adapter_params(self):
        p = list(self.encoder.parameters()) + list(self.workspace.parameters())
        p += list(self.xattn.parameters()) + [self.gate]
        return p

    def gate_str(self):
        return f"xattn_gate={torch.tanh(self.gate).item():.3f}"


# -------------------- Training & Eval --------------------

def train_backbone(model, data, steps=800, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        seq = data[step % len(data)].unsqueeze(0)
        logits = model(seq)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 200 == 0:
            print(f"    Step {step+1:4d} | loss: {loss.item():.4f}")


def train_adapted(model, data, steps=600, lr=3e-3):
    opt = torch.optim.AdamW(model.adapter_params(), lr=lr)
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.train()
    for step in range(steps):
        prompt, seq = data[step % len(data)]
        logits = model(seq.unsqueeze(0), prompt.unsqueeze(0))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), seq[1:].unsqueeze(0).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 150 == 0:
            print(f"    Step {step+1:4d} | loss: {loss.item():.4f} | {model.gate_str()}")


def check_pattern(tokens, motif, start=1):
    correct = total = 0
    for i in range(start, len(tokens)):
        if tokens[i] == motif[(i - start) % len(motif)]:
            correct += 1
        total += 1
    return correct, total


def eval_redirect(model, n=50):
    model.backbone.eval()
    c = t = 0
    for _ in range(n):
        src = torch.randint(0, 5, (1,)).item()
        tgt = (src + 1) % 5
        prompt = torch.tensor([[tgt + 10]], device=DEVICE)
        start = torch.tensor([[src + 10]], device=DEVICE)
        tokens = model.generate(prompt, start, n_tokens=16)
        ci, ti = check_pattern(tokens, PATTERNS[tgt])
        c += ci; t += ti
    return c / max(1, t)


def eval_same_class(model, n=50):
    model.backbone.eval()
    c = t = 0
    for cls in range(5):
        for _ in range(n // 5):
            prompt = torch.tensor([[cls + 10]], device=DEVICE)
            start = torch.tensor([[cls + 10]], device=DEVICE)
            tokens = model.generate(prompt, start, n_tokens=16)
            ci, ti = check_pattern(tokens, PATTERNS[cls])
            c += ci; t += ti
    return c / max(1, t)


# -------------------- Main --------------------

def test_approach(name, model_cls, backbone_data, redir_data, **kwargs):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    backbone = TinyLM().to(DEVICE)
    print("  Pretraining backbone...")
    train_backbone(backbone, backbone_data)
    acc = eval_same_class(type('M', (), {
        'backbone': backbone,
        'generate': lambda self, p, s, n_tokens=16: backbone.generate(s, n_tokens)
    })(), 20)
    print(f"  Backbone accuracy: {acc:.1%}")

    print(f"  Adding {name} adapters...")
    model = model_cls(backbone, **kwargs)

    print("  Training adapters...")
    train_adapted(model, redir_data)

    redir = eval_redirect(model)
    same = eval_same_class(model)
    print(f"\n  REDIRECT accuracy:   {redir:.1%}")
    print(f"  SAME-CLASS accuracy: {same:.1%}")
    print(f"  Gates: {model.gate_str()}")
    return redir, same


def main():
    torch.manual_seed(42)
    print("=" * 60)
    print("  Duplex-1.2 Architecture Comparison Test")
    print(f"  Device: {DEVICE} | Backbone: {N_LAYERS} layers, {D_MODEL}d")
    print("=" * 60)

    backbone_data = make_pattern_data(500)
    redir_data = make_redirect_data(500)

    results = {}

    # A: Cross-attention (current approach)
    r_a, s_a = test_approach("A: Cross-Attention Adapters (sparse, 3/6 layers)",
                              CrossAttnModel, backbone_data, redir_data, interval=2)
    results["xattn"] = (r_a, s_a)

    # B: Prefix conditioning (proposed)
    r_b, s_b = test_approach("B: Prefix Conditioning (8 soft tokens)",
                              PrefixModel, backbone_data, redir_data, n_prefix=8)
    results["prefix"] = (r_b, s_b)

    # C: Hybrid
    r_c, s_c = test_approach("C: Hybrid (Prefix + 1 cross-attn at last layer)",
                              HybridModel, backbone_data, redir_data, n_prefix=8)
    results["hybrid"] = (r_c, s_c)

    # VERDICT
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Approach':<45} {'Redirect':>10} {'Same-cls':>10}")
    print(f"  {'-'*65}")
    print(f"  {'A: Cross-Attention (sparse)':<45} {r_a:>9.1%} {s_a:>9.1%}")
    print(f"  {'B: Prefix Conditioning':<45} {r_b:>9.1%} {s_b:>9.1%}")
    print(f"  {'C: Hybrid (Prefix + 1 xattn)':<45} {r_c:>9.1%} {s_c:>9.1%}")
    print()

    best_name, best_r = max(
        [("A: Cross-Attention", r_a), ("B: Prefix", r_b), ("C: Hybrid", r_c)],
        key=lambda x: x[1]
    )
    print(f"  BEST: {best_name} with {best_r:.1%} redirect accuracy")

    if best_r >= 0.5:
        print(f"  VERDICT: PASS — ready for H200 training.")
    elif best_r >= 0.3:
        print(f"  VERDICT: PARTIAL — promising but needs tuning.")
    else:
        print(f"  VERDICT: FAIL — more architecture work needed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
