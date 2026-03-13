# Duplex-1.4-1.7B: Full-Duplex Language Model

A language model that can **accept input while it's outputting** — built by adding deep prefix conditioning (P-Tuning v2) to a frozen Qwen3-1.7B-Base.

## Model Naming

- **Duplex-1.4-1.7B** — fourth major iteration, 1.7B base size
- Future: Duplex-2-7B, Duplex-2-70B, etc.

## Architecture

**Baseline (Qwen3-1.7B):** Standard decoder-only transformer. Half-duplex — you either provide input or get output, never both at once. To correct the model mid-response, you must stop it, re-prompt, and restart.

**Duplex-1.4-1.7B:** Same Qwen3-1.7B backbone (**fully frozen**), plus three trainable components:
- **Update Encoder** (4-layer bidirectional transformer, ~155M params): encodes prompt/correction text into per-token representations
- **Workspace Module** (32 latent slots × 2048 dim, ~95M params): compresses context into latent slots via cross-attention pooling with gated updates
- **Deep Prefix Encoder** (per-layer K/V projections, ~115M params): projects workspace slots into key-value pairs injected at every Qwen layer

**Key design: deep prefix conditioning (P-Tuning v2).** Workspace slots are projected into per-layer K/V pairs and injected at all 28 Qwen attention layers via `past_key_values`. Each layer directly sees the workspace — no signal attenuation through frozen layers.

### Why deep prefix, not shallow prefix?

v1.3 used shallow prefix (prepend to input embeddings only). The prefix signal entered at layer 0 and had to survive 28 frozen attention layers. By the deeper layers, the signal was diluted to nothing — a known failure mode for 1B-3B models documented in the P-Tuning v2 paper (Liu et al., 2022).

Deep prefix injects fresh K/V at **every layer**, giving direct influence at every depth:

| Approach | Prefix steering | Redirect | Correction |
|----------|:-:|:-:|:-:|
| Cross-attention adapters (v1.0-1.1) | 0% | 0% | N/A |
| Shallow prefix (v1.2-1.3) | 100% local / 0% Qwen | 100% local / 0% Qwen | N/A |
| **Deep prefix (v1.4)** | **100%** | **100%** | **100%** |

### Revision markers

Corrections use **existing Qwen tokens** (`[[REVISE:`, `]]`) rather than new special tokens. This keeps Qwen's embedding table and output head fully frozen — no new token learning required.

## How It Works

**Standard LLMs** have one conditioning channel — the token sequence:
```
p(y_t | y_<t) = softmax(Head(Decoder(Embed(y_<t))))
```
New information can only enter by rewriting the sequence and restarting.

**Duplex** adds a second channel — workspace-derived K/V injected at every layer:
```
K_layer = [DeepPrefix_K(W); SelfAttn_K(y_<t)]
V_layer = [DeepPrefix_V(W); SelfAttn_V(y_<t)]
p(y_t | y_<t, W) = softmax(Head(Decoder(y_<t, K_layer, V_layer)))
```

When a correction arrives mid-generation:
1. Encode correction: `e = Encoder(correction_tokens)`
2. Update workspace: `W' = GatedUpdate(W, CrossAttnPool(W, e))`
3. Reproject prefix: K/V pairs at all 28 layers update instantly
4. Next token sees updated K/V through attention at every layer → generation shifts

The decoder's next-token distribution changes because the K/V changed at every layer, not because the token sequence was rewritten. That's full-duplex: input updates the conditioning state while output continues uninterrupted.

### Training

- **Qwen 1.7B**: fully frozen (all 1.7B params)
- **Trainable**: encoder + workspace + deep prefix (~365M params, ~19% of total)
- **Token dropout** (50%): randomly corrupts response tokens in the decoder input during training, forcing the model to rely on the prefix for task-specific information instead of teacher-forced text context
- **Two-phase training**: Phase 1 (workspace-conditioned generation), Phase 2 (mid-stream correction)

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
    checkpoint_path="checkpoints/duplex-1.4-1.7b/phase2_best.pt",
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

The prompt is encoded through the workspace and projected into per-layer K/V pairs. The decoder receives a generic instruction internally — the model gets all task-specific info from the deep prefix, not the decoder input.

### Mid-stream correction (full-duplex)

```python
response, text_at_correction = model.generate_with_update(
    prompt_text="Write a short bio for James, a 28-year-old chef from Paris.",
    max_new_tokens=200,
    temperature=0.7,
    correction_text="Update: the person's name is Marco, not James.",
    correction_after_tokens=15,
)
print(response)              # should reflect "Marco" instead of "James"
print(text_at_correction)    # what was generated before the correction hit
```

What happens internally:
1. Encoder processes the prompt -> workspace built -> deep prefix K/V at all 28 layers
2. Model starts generating (decoder sees generic instruction, attention sees prefix K/V)
3. At token 15, correction is encoded, workspace updates, K/V at all layers refresh
4. Subsequent tokens are conditioned on updated K/V at every layer
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
    print(text, end="\r")
```

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
torchrun --nproc_per_node=2 scripts/train.py --phase 2 --max_steps 5000 --batch_size 8 --grad_accum 16 --resume checkpoints/duplex-1.4-1.7b/phase1_best.pt
```

### 4. Evaluate
```bash
python scripts/check_phase2.py       # quick PASS/FAIL on 4 correction scenarios
python scripts/evaluate.py --duplex_ckpt checkpoints/duplex-1.4-1.7b/final.pt --n_samples 200
```

### 5. Demo
```bash
python scripts/demo.py --duplex_ckpt checkpoints/duplex-1.4-1.7b/final.pt   # side-by-side: Qwen vs Duplex
python scripts/demo2.py               # ChatGPT-style single-model demo
```

### 6. Local architecture test (no Qwen download needed)
```bash
python scripts/test_local_final.py    # validates deep prefix conditioning on a tiny model
```

## GPU Requirements

| Setup | Batch size | Grad accum | Steps/s | Phase 1 time |
|-------|-----------|------------|---------|--------------|
| 1x A100 80GB | 8 | 16 | ~0.10 | ~5.5 hrs |
| 2x A100 80GB | 8 | 16 | ~0.19 | ~3 hrs |
| 2x H200 141GB | 16 | 8 | ~0.37 | ~1.5 hrs |

Batch size 32 per GPU OOMs on A100 80GB. Use batch_size 8 with gradient accumulation.

## Project Structure

```
duplex/
    config.py              -- Model config, revision markers
    encoder.py             -- Update Encoder (4-layer bidirectional transformer)
    workspace.py           -- Workspace module (32 slots, gated update)
    duplex_model.py        -- Main model: frozen Qwen + deep prefix (P-Tuning v2)
    renderer.py            -- Interprets revision markers for display
    data/
        tasks.py           -- 7 synthetic task generators
        dataset.py         -- Training dataset (with token dropout)
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
    test_local_final.py    -- Local deep prefix validation
    demo.py                -- Gradio investor demo (side-by-side streaming)
    demo2.py               -- ChatGPT-style single-model demo
    demo3.py               -- Breakthrough-focused preset demo
CONCEPT1/                  -- Original toy prototype (proof of concept)
```

## Architecture Evolution

| Version | Architecture | Result |
|---------|-------------|--------|
| 1.0 | Cross-attention adapters (28 layers), zero-init | Dead gradients — both zeros killed all learning |
| 1.1 | Cross-attention adapters, small init + tanh gates | Cascade corruption — 28 layers of perturbation -> gibberish |
| 1.2 | Shallow prefix conditioning (input embeddings only) | Works locally, fails at Qwen scale — signal attenuates through 28 frozen layers |
| 1.3 | Shallow prefix + existing token markers + token dropout + generic instruction | Same attenuation problem — prefix ignored by Qwen |
| **1.4** | **Deep prefix (P-Tuning v2) — per-layer K/V injection** | **100% prefix steering, 100% redirect, 100% correction locally** |

## Prior Work (CONCEPT1)

The toy prototype in `CONCEPT1/` validated the core idea at small scale (2M baseline vs 5M workspace model). Key ablation result: disabling the workspace update dropped revision accuracy from 55% to 4.4% and raised contradiction rate from 3.6% to 40.2%. This confirmed the full-duplex mechanism works and justified scaling to Duplex-1.4-1.7B.
