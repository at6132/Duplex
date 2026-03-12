# Duplex-1.3-1.7B: Full-Duplex Language Model

A language model that can **accept input while it's outputting** — built by adding prefix-based conditioning to a frozen Qwen3-1.7B-Base.

## Model Naming

- **Duplex-1.3-1.7B** — third iteration, 1.7B base size
- Future: Duplex-2-4B, Duplex-2-8B, etc.

## Architecture

**Baseline (Qwen3-1.7B):** Standard decoder-only transformer. Half-duplex — you either provide input or get output, never both at once. To correct the model mid-response, you must stop it, re-prompt, and restart.

**Duplex-1.3-1.7B:** Same Qwen3-1.7B backbone (**fully frozen**), plus two trainable components:
- **Update Encoder** (4-layer bidirectional transformer, ~50M params): encodes prompt/correction text into per-token representations
- **Workspace Module** (32 latent slots × 2048 dim, ~200M params): compresses context into soft prefix tokens via cross-attention pooling with gated updates

**Key design: prefix conditioning.** Workspace slots are prepended as soft tokens directly to Qwen's input embeddings. Qwen's own self-attention naturally conditions on them — no monkey-patching, no hidden-state perturbation, no cross-attention adapters injected into decoder layers.

### Why prefix, not cross-attention adapters?

Versions 1.0–1.1 used cross-attention adapters injected into every (or every Nth) decoder layer. These caused **cascading perturbation** during autoregressive generation: even small per-layer noise compounded across 28 layers, producing gibberish at inference despite good training loss.

Local testing confirmed:
| Approach | Redirect Accuracy | Same-Class |
|----------|:-:|:-:|
| Cross-attention adapters (all 28 layers) | 0% | 100% |
| Cross-attention adapters (sparse, 7 layers) | 0% | 100% |
| **Prefix conditioning** | **100%** | **100%** |

The prefix approach works because the backbone's own self-attention handles conditioning — it already knows how to attend to input tokens.

### Revision markers

Corrections use **existing Qwen tokens** (`[[REVISE:`, `]]`) rather than new special tokens. This keeps Qwen's embedding table and output head fully frozen — no new token learning required.

## How It Works

**Standard LLMs** have one conditioning channel — the token sequence:
```
p(y_t | y_<t) = softmax(Head(Decoder(Embed(y_<t))))
```
New information can only enter by rewriting the sequence and restarting.

**Duplex** adds a second channel — the workspace prefix **W**:
```
p(y_t | y_<t, W) = softmax(Head(Decoder([W; Embed(y_<t)])))
```

When a correction arrives mid-generation:
1. Encode correction: `e = Encoder(correction_tokens)`
2. Update workspace: `W' = GatedUpdate(W, CrossAttnPool(W, e))`
3. Extend prefix: `prefix = [W'; e]` (workspace + raw correction tokens)
4. Next token sees updated prefix through self-attention → generation shifts

The decoder's next-token distribution changes because the prefix changed, not because the token sequence was rewritten. That's full-duplex: input updates the conditioning state while output continues uninterrupted.

### Training

- **Qwen 1.7B**: fully frozen (all 1.7B params)
- **Trainable**: encoder + workspace (~250M params, ~13% of total)
- **Joint training**: all tasks (workspace-conditioned generation + mid-stream correction) trained simultaneously to avoid catastrophic forgetting between phases

## Setup

```bash
pip install -r requirements.txt
python scripts/download_qwen.py   # or: download Qwen3-1.7B-Base to models/qwen3-1.7b-base
```

## Using the Model

### Load a trained checkpoint

```python
from duplex.inference.generate import load_duplex_model

model = load_duplex_model(
    qwen_path="models/qwen3-1.7b-base",
    checkpoint_path="checkpoints/duplex-1.3-1.7b/phase2_best.pt",
)
model.eval()
```

### Basic generation (no correction)

```python
response, _ = model.generate_with_update(
    prompt_text="Write a short bio for James, a 28-year-old chef from Paris.",
    max_new_tokens=200,
    temperature=0.7,
)
print(response)
```

The prompt is encoded through the workspace prefix. The decoder receives a generic instruction internally — the model gets all task-specific info from the prefix, not the decoder input.

### Mid-stream correction (full-duplex)

```python
response, text_at_correction = model.generate_with_update(
    prompt_text="Write a short bio for James, a 28-year-old chef from Paris.",
    max_new_tokens=200,
    temperature=0.7,
    correction_text="Update: the person's name is Marco, not James.",
    correction_after_tokens=15,  # inject correction after 15 generated tokens
)
print(response)              # should reflect "Marco" instead of "James"
print(text_at_correction)    # what was generated before the correction hit
```

What happens internally:
1. Encoder processes the prompt -> workspace prefix is built
2. Model starts generating (decoder sees generic instruction + prefix)
3. At token 15, the correction is encoded and the workspace updates
4. The prefix changes — subsequent tokens are conditioned on the updated workspace
5. No stop, no restart. Generation continues with new information.

### Streaming generation

```python
for text in model.generate_with_update_streaming(
    prompt_text="Describe a tourist visiting New York for the first time.",
    max_new_tokens=200,
    temperature=0.7,
    correction_text="Change the city to Tokyo.",
    correction_after_tokens=12,
):
    print(text, end="\r")  # overwrite line for live streaming effect
```

The stream yields the current full response text after each token. When the correction is injected, it yields a visual injection marker before continuing.

## Training from Scratch

### 1. Generate training data
```bash
python scripts/generate_data.py --n_train 500000 --n_val 20000 --n_test 10000
```

### 2. Train Phase 1 (workspace-conditioned generation)
```bash
# Multi-GPU (adjust --nproc_per_node to your GPU count)
torchrun --nproc_per_node=2 scripts/train.py --phase 1 --max_steps 2000 --batch_size 8 --grad_accum 16

# Single GPU
python scripts/train.py --phase 1 --max_steps 2000
```

### 3. Train Phase 2 (mid-stream correction)
```bash
torchrun --nproc_per_node=2 scripts/train.py --phase 2 --max_steps 5000 --batch_size 8 --grad_accum 16 --resume checkpoints/duplex-1.3-1.7b/phase1_best.pt
```

### 4. Evaluate
```bash
python scripts/check_phase2.py       # quick PASS/FAIL on 4 correction scenarios
python scripts/evaluate.py --duplex_ckpt checkpoints/duplex-1.3-1.7b/final.pt --n_samples 200
```

### 5. Demo
```bash
python scripts/demo.py --duplex_ckpt checkpoints/duplex-1.3-1.7b/final.pt   # side-by-side: Qwen vs Duplex
python scripts/demo2.py               # ChatGPT-style single-model demo
```

### 6. Local architecture test (no Qwen download needed)
```bash
python scripts/test_local_final.py    # validates prefix conditioning on a tiny model
```

## GPU Requirements

| Setup | Batch size | Grad accum | Steps/s | Phase 1 time |
|-------|-----------|------------|---------|--------------|
| 1x A100 80GB | 8 | 16 | ~0.10 | ~5.5 hrs |
| 2x A100 80GB | 8 | 16 | ~0.19 | ~3 hrs |
| 2x H200 141GB | 16 | 8 | ~0.37 | ~1.5 hrs |

Batch size 32 per GPU OOMs on A100 80GB. Use batch_size 8 with gradient accumulation to maintain the same effective batch size.

## Project Structure

```
duplex/
    config.py              -- Model config, revision markers
    encoder.py             -- Update Encoder (4-layer bidirectional transformer)
    workspace.py           -- Workspace module (32 slots, gated update)
    duplex_model.py        -- Main model: frozen Qwen + prefix conditioning
    renderer.py            -- Interprets revision markers for display
    data/
        tasks.py           -- 7 synthetic task generators
        dataset.py         -- Training dataset
    training/
        trainer.py         -- Training loop (DDP, EMA convergence, checkpointing)
    inference/
        generate.py        -- Generation with mid-stream workspace update
scripts/
    download_qwen.py       -- Download Qwen3-1.7B-Base
    generate_data.py       -- Generate synthetic training data
    train.py               -- Training CLI (single/multi-GPU)
    evaluate.py            -- Evaluation CLI
    check_phase2.py        -- Quick phase 2 verification (PASS/FAIL)
    test_local.py          -- Local architecture test (CPU/GPU, no Qwen needed)
    test_local_final.py    -- Comprehensive local stress test
    demo.py                -- Gradio investor demo (side-by-side streaming)
    demo2.py               -- ChatGPT-style single-model demo
    demo3.py               -- Breakthrough-focused preset demo
CONCEPT1/                  -- Original toy prototype (proof of concept)
```

## Architecture Evolution

| Version | Architecture | Problem |
|---------|-------------|---------|
| 1.0 | Cross-attention adapters (28 layers), zero-init o_proj + zero-init gates | Dead gradients — both zeros killed all learning |
| 1.1 | Cross-attention adapters, small o_proj init + tanh gates | Cascade corruption — 28 layers of perturbation → gibberish at inference |
| 1.2 | Sparse adapters (every 4th layer) + RMSNorm output | Still corrupted; adapters too weak to redirect, too strong to be harmless |
| **1.3** | **Prefix conditioning (no adapters)** | **Works: 100% redirect, 100% same-class, Qwen fully frozen** |

## Prior Work (CONCEPT1)

The toy prototype in `CONCEPT1/` validated the core idea at small scale (2M baseline vs 5M workspace model). Key ablation result: disabling the workspace update dropped revision accuracy from 55% to 4.4% and raised contradiction rate from 3.6% to 40.2%. This confirmed the full-duplex mechanism works and justified scaling to Duplex-1.3-1.7B.
