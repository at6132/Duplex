"""
Local validation for Duplex-1.4 deep prefix conditioning (P-Tuning v2).

KEY CHANGE from v1.3: instead of prepending workspace slots to
input embeddings (which gets diluted through 28 frozen layers), we inject
workspace-derived K/V pairs at EVERY backbone layer via past_key_values.

Tests:
  1. Backbone sanity: class token start -> correct motif (no adapter)
  2. Prefix-only steering: GENERIC start + deep prefix -> correct motif
  3. Redirect: wrong class start + deep prefix -> prefix's motif wins
  4. Correction prefix: prefix(src+correction) -> output switches to target
  5. Mid-stream correction: inject correction at step 6 -> output switches
  6. No correction -> stays on original class

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
HEAD_DIM = D_MODEL // N_HEADS
N_LAYERS = 12
FF_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CLASSES = 5
MOTIFS = {k: [4*k, 4*k+1, 4*k+2, 4*k+3] for k in range(N_CLASSES)}
GENERIC = 25
SEQ_LEN = 24
N_PREFIX = 12


# -------------------- Backbone --------------------

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x, prefix_kv=None):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, N_HEADS, HEAD_DIM)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if prefix_kv is not None:
            pk, pv = prefix_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        T_k = k.size(2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        # Causal mask: each query position can attend to all K positions up to
        # (prefix_len + query_position). prefix_kv positions are always visible.
        P_len = T_k - T
        causal = torch.ones(T, T_k, device=x.device, dtype=torch.bool)
        for i in range(T):
            causal[i, P_len + i + 1:] = False
        scores.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        return self.out((F.softmax(scores, -1) @ v).transpose(1, 2).contiguous().view(B, T, D_MODEL))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = Attn()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, FF_DIM), nn.GELU(), nn.Linear(FF_DIM, D_MODEL))

    def forward(self, x, prefix_kv=None):
        x = x + self.attn(self.ln1(x), prefix_kv=prefix_kv)
        x = x + self.ff(self.ln2(x))
        return x


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward_embeds(self, e, prefix_kvs=None):
        for i, layer in enumerate(self.layers):
            kv = prefix_kvs[i] if prefix_kvs is not None else None
            e = layer(e, prefix_kv=kv)
        return self.head(self.ln_f(e))

    def forward(self, ids, prefix_kvs=None):
        return self.forward_embeds(self.embed(ids), prefix_kvs=prefix_kvs)


# -------------------- Deep Prefix Model --------------------

class DeepPrefixLM(nn.Module):
    def __init__(self, backbone, n_prefix=N_PREFIX):
        super().__init__()
        self.backbone = backbone
        self.n_prefix = n_prefix
        self.encoder = UpdateEncoder(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            d_ff=FF_DIM, n_layers=2, dropout=0.0).to(DEVICE)
        self.workspace = WorkspaceModule(
            n_slots=n_prefix, d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM,
            dropout=0.0).to(DEVICE)
        kv_dim = N_HEADS * HEAD_DIM
        self.kv_projs = nn.ModuleList([
            nn.Linear(D_MODEL, 2 * kv_dim, bias=False)
            for _ in range(N_LAYERS)
        ]).to(DEVICE)

    def build_workspace(self, prompt_ids, correction_ids=None):
        enc = self.encoder(prompt_ids)
        ws = self.workspace(enc)
        if correction_ids is not None:
            corr_enc = self.encoder(correction_ids)
            ws = self.workspace(corr_enc, workspace=ws)
        return ws

    def build_prefix_kvs(self, workspace):
        """Build per-layer (K, V) tuples from workspace."""
        B, P, _ = workspace.shape
        kvs = []
        for layer_idx in range(N_LAYERS):
            kv = self.kv_projs[layer_idx](workspace)
            k, v = kv.chunk(2, dim=-1)
            k = k.view(B, P, N_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(B, P, N_HEADS, HEAD_DIM).transpose(1, 2)
            kvs.append((k, v))
        return kvs

    def forward_with_prefix(self, workspace, input_ids):
        prefix_kvs = self.build_prefix_kvs(workspace)
        embs = self.backbone.embed(input_ids)
        logits = self.backbone.forward_embeds(embs, prefix_kvs=prefix_kvs)
        return logits

    @torch.no_grad()
    def generate(self, workspace, start_ids, n_tokens=20):
        ids = start_ids.clone()
        prefix_kvs = self.build_prefix_kvs(workspace)
        for _ in range(n_tokens):
            embs = self.backbone.embed(ids)
            logits = self.backbone.forward_embeds(embs, prefix_kvs=prefix_kvs)
            ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
        return ids[0].tolist()

    @torch.no_grad()
    def generate_with_correction(self, prompt_ids, start_ids, correction_ids,
                                  inject_at, n_tokens=20):
        ws = self.build_workspace(prompt_ids)
        ids = start_ids.clone()
        for step in range(n_tokens):
            if step == inject_at:
                ws = self.build_workspace(prompt_ids, correction_ids)
            prefix_kvs = self.build_prefix_kvs(ws)
            embs = self.backbone.embed(ids)
            logits = self.backbone.forward_embeds(embs, prefix_kvs=prefix_kvs)
            ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
        return ids[0].tolist()

    def adapter_params(self):
        return (list(self.encoder.parameters())
                + list(self.workspace.parameters())
                + list(self.kv_projs.parameters()))


# -------------------- Training Data --------------------

def make_motif_seq(cls, length=SEQ_LEN):
    motif = MOTIFS[cls]
    return (motif * ((length) // len(motif) + 1))[:length]


def sample_task():
    task = random.randint(0, 2)
    if task == 0:
        cls = random.randint(0, N_CLASSES-1)
        prompt = torch.tensor([cls + 20], device=DEVICE)
        target = torch.tensor([GENERIC] + make_motif_seq(cls, SEQ_LEN-1), device=DEVICE)
        return "prefix_steer", prompt, None, target
    elif task == 1:
        src = random.randint(0, N_CLASSES-1)
        tgt = (src + random.randint(1, N_CLASSES-1)) % N_CLASSES
        prompt = torch.tensor([tgt + 20], device=DEVICE)
        target = torch.tensor([src + 20] + make_motif_seq(tgt, SEQ_LEN-1), device=DEVICE)
        return "redirect", prompt, None, target
    else:
        src = random.randint(0, N_CLASSES-1)
        tgt = (src + random.randint(1, N_CLASSES-1)) % N_CLASSES
        prompt = torch.tensor([src + 20], device=DEVICE)
        correction = torch.tensor([tgt + 20], device=DEVICE)
        target = torch.tensor([GENERIC] + make_motif_seq(tgt, SEQ_LEN-1), device=DEVICE)
        return "correction", prompt, correction, target


# -------------------- Evaluation Helpers --------------------

def check_motif(tokens, motif, start=1, length=20):
    correct = total = 0
    for i in range(start, min(start + length, len(tokens))):
        if tokens[i] == motif[(i - start) % len(motif)]:
            correct += 1
        total += 1
    return correct / max(1, total)


def classify_output(tokens, start=1, length=10):
    best_cls, best_score = -1, -1
    for cls in range(N_CLASSES):
        score = check_motif(tokens, MOTIFS[cls], start, length)
        if score > best_score:
            best_score = score
            best_cls = cls
    return best_cls, best_score


def main():
    torch.manual_seed(42)
    random.seed(42)

    print("=" * 64)
    print("  Duplex-1.4 Deep Prefix Local Validation (P-Tuning v2)")
    print(f"  Device: {DEVICE}")
    print()
    print(f"  KEY: K/V injected at EVERY layer (not just input embeddings)")
    print(f"  GENERIC start token ({GENERIC}), prefix = class info")
    print(f"  Backbone: {N_LAYERS}L, {D_MODEL}d  |  {N_PREFIX} prefix slots")
    print(f"  Training: ALL tasks jointly")
    print("=" * 64)

    # Phase 1: pretrain backbone
    print("\n[BACKBONE] Pretraining on class->motif patterns...")
    backbone = LM().to(DEVICE)
    opt_bb = torch.optim.AdamW(backbone.parameters(), lr=1e-3)
    backbone.train()
    for step in range(1500):
        cls = random.randint(0, N_CLASSES-1)
        seq = torch.tensor(
            [cls + 20] + make_motif_seq(cls, SEQ_LEN-1), device=DEVICE
        ).unsqueeze(0)
        logits = backbone(seq)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        opt_bb.zero_grad(); loss.backward(); opt_bb.step()
        if (step + 1) % 500 == 0:
            print(f"    Step {step+1:4d} | loss: {loss.item():.4f}")

    backbone.eval()
    with torch.no_grad():
        for tc in range(min(3, N_CLASSES)):
            ids = torch.tensor([[tc + 20]], device=DEVICE)
            for _ in range(8):
                logits = backbone(ids)
                ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
            print(f"    Sanity class {tc}: {ids[0].tolist()}")

    # Phase 2: joint adapter training with DEEP prefix
    print(f"\n[ADAPTER] Joint training with DEEP prefix (K/V at every layer)")
    model = DeepPrefixLM(backbone, n_prefix=N_PREFIX).to(DEVICE)
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(model.adapter_params(), lr=1e-3)
    model.train()

    N_STEPS = 8000
    task_counts = {"prefix_steer": 0, "redirect": 0, "correction": 0}
    for step in range(N_STEPS):
        task_type, prompt, correction, target = sample_task()
        task_counts[task_type] += 1

        ws = model.build_workspace(prompt.unsqueeze(0),
                                    correction.unsqueeze(0) if correction is not None else None)
        logits = model.forward_with_prefix(ws, target.unsqueeze(0))
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), target[1:].unsqueeze(0).reshape(-1)
        )
        opt.zero_grad(); loss.backward(); opt.step()

        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1:4d} | loss: {loss.item():.4f} | tasks: {task_counts}")

    # ---- EVALUATION ----
    print(f"\n{'='*64}")
    print("  EVALUATION")
    print(f"{'='*64}")
    model.backbone.eval()
    N_EVAL = 80

    # Test 1: Backbone sanity
    print("\n  [1] Backbone sanity (class token -> motif, no prefix)...")
    bb_scores = []
    with torch.no_grad():
        for _ in range(N_EVAL):
            cls = random.randint(0, N_CLASSES-1)
            ids = torch.tensor([[cls + 20]], device=DEVICE)
            for _ in range(20):
                logits = backbone(ids)
                ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
            bb_scores.append(check_motif(ids[0].tolist(), MOTIFS[cls]))
    bb = sum(bb_scores) / len(bb_scores)
    print(f"      {bb:.1%}")

    # Test 2: Prefix-only steering
    print("  [2] Deep prefix steering (GENERIC start, prefix = class)...")
    ps_scores = []
    for _ in range(N_EVAL):
        cls = random.randint(0, N_CLASSES-1)
        ws = model.build_workspace(torch.tensor([[cls + 20]], device=DEVICE))
        tokens = model.generate(ws, torch.tensor([[GENERIC]], device=DEVICE), 20)
        ps_scores.append(check_motif(tokens, MOTIFS[cls]))
    ps = sum(ps_scores) / len(ps_scores)
    print(f"      {ps:.1%}")

    # Test 3: Redirect
    print("  [3] Redirect (start=src, prefix=tgt -> tgt motif)...")
    rd_scores = []
    for _ in range(N_EVAL):
        src = random.randint(0, N_CLASSES-1)
        tgt = (src + 1) % N_CLASSES
        ws = model.build_workspace(torch.tensor([[tgt + 20]], device=DEVICE))
        tokens = model.generate(ws, torch.tensor([[src + 20]], device=DEVICE), 20)
        rd_scores.append(check_motif(tokens, MOTIFS[tgt]))
    rd = sum(rd_scores) / len(rd_scores)
    print(f"      {rd:.1%}")

    # Test 4: Correction prefix
    print("  [4] Correction prefix (src + tgt correction -> tgt motif)...")
    cr_scores = []
    for _ in range(N_EVAL):
        src = random.randint(0, N_CLASSES-1)
        tgt = (src + 1) % N_CLASSES
        ws = model.build_workspace(
            torch.tensor([[src + 20]], device=DEVICE),
            torch.tensor([[tgt + 20]], device=DEVICE),
        )
        tokens = model.generate(ws, torch.tensor([[GENERIC]], device=DEVICE), 20)
        cr_scores.append(check_motif(tokens, MOTIFS[tgt]))
    cr = sum(cr_scores) / len(cr_scores)
    print(f"      {cr:.1%}")

    # Test 5: Mid-stream correction
    print("  [5] Mid-stream correction (inject at step 6 -> output switches)...")
    ms_scores = []
    ms_details = {"pre_correct": 0.0, "post_correct": 0.0}
    for _ in range(N_EVAL):
        src = random.randint(0, N_CLASSES-1)
        tgt = (src + 1) % N_CLASSES
        tokens = model.generate_with_correction(
            torch.tensor([[src + 20]], device=DEVICE),
            torch.tensor([[GENERIC]], device=DEVICE),
            torch.tensor([[tgt + 20]], device=DEVICE),
            inject_at=6, n_tokens=20,
        )
        pre = check_motif(tokens, MOTIFS[src], start=1, length=6)
        post_cls, post_score = classify_output(tokens, start=8, length=12)
        switched = 1.0 if post_cls == tgt else 0.0
        ms_scores.append(switched)
        ms_details["pre_correct"] += pre
        ms_details["post_correct"] += post_score
    ms = sum(ms_scores) / len(ms_scores)
    ms_details = {k: v / N_EVAL for k, v in ms_details.items()}
    print(f"      Switch rate: {ms:.1%}")
    print(f"      Pre-correction motif accuracy:  {ms_details['pre_correct']:.1%}")
    print(f"      Post-correction motif accuracy: {ms_details['post_correct']:.1%}")

    # Test 6: No correction -> stays on original class
    print("  [6] No correction -> stays on prompted class...")
    nc_scores = []
    for _ in range(N_EVAL):
        cls = random.randint(0, N_CLASSES-1)
        ws = model.build_workspace(torch.tensor([[cls + 20]], device=DEVICE))
        tokens = model.generate(ws, torch.tensor([[GENERIC]], device=DEVICE), 20)
        nc_scores.append(check_motif(tokens, MOTIFS[cls]))
    nc = sum(nc_scores) / len(nc_scores)
    print(f"      {nc:.1%}")

    # VERDICT
    print(f"\n{'='*64}")
    print("  FINAL SCORECARD -- Duplex-1.4 Deep Prefix")
    print(f"{'='*64}")
    print(f"  {'Test':<55} {'Score':>8}")
    print(f"  {'-'*65}")
    print(f"  {'1. Backbone sanity (no prefix)':<55} {bb:>7.1%}")
    print(f"  {'2. Deep prefix steering (GENERIC start) [KEY]':<55} {ps:>7.1%}")
    print(f"  {'3. Redirect (prefix overrides start token)':<55} {rd:>7.1%}")
    print(f"  {'4. Correction prefix -> switches to target':<55} {cr:>7.1%}")
    print(f"  {'5. Mid-stream correction -> output switches':<55} {ms:>7.1%}")
    print(f"  {'6. No correction -> stays on original class':<55} {nc:>7.1%}")
    print()

    arch_tests = [ps, rd]
    arch_avg = sum(arch_tests) / len(arch_tests)
    correction_tests = [cr, ms]
    corr_avg = sum(correction_tests) / len(correction_tests)

    print(f"  Architecture avg (Tests 2-3):  {arch_avg:.1%}")
    print(f"  Correction avg (Tests 4-5):    {corr_avg:.1%}")
    print()

    if arch_avg >= 0.9 and corr_avg >= 0.5:
        print("  VERDICT: PASS")
        print("  Deep prefix conditioning validated + corrections working.")
        print("  Safe to train on H200s / A100s.")
    elif arch_avg >= 0.9:
        print("  VERDICT: ARCHITECTURE PASS")
        print("  Deep prefix conditioning works perfectly.")
        if corr_avg >= 0.3:
            print(f"  Correction behavior emerging ({corr_avg:.0%}).")
        else:
            print(f"  Correction behavior weak ({corr_avg:.0%}) in toy model.")
        print("  Safe to deploy Phase 1 to GPUs.")
    elif arch_avg >= 0.7:
        print("  VERDICT: PARTIAL")
        print("  Architecture works but needs tuning.")
    else:
        print("  VERDICT: FAIL")
        print("  Architecture needs more work. Do NOT deploy.")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
