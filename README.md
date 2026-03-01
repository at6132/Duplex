# Duplex-1.1-1.7B: Full-Duplex Language Model

A language model that can **accept input while it's outputting** -- built by adding workspace adapter layers to a frozen Qwen3-1.7B-Base.

## Model Naming

Following standard LLM conventions:
- **Duplex-1.1-1.7B** -- first generation v1.1, 1.7B base size
- Future: Duplex-1-4B, Duplex-1-8B, Duplex-2-X, etc.

## Architecture

**Baseline (Qwen3-1.7B):** Standard decoder-only transformer. Half-duplex -- you either provide input or get output, never both at once. To correct the model mid-response, you must stop it, re-prompt, and restart.

**Duplex-1.1-1.7B:** Same Qwen3-1.7B backbone (frozen), plus three new trainable components:
- **Update Encoder** (4-layer bidirectional transformer): encodes prompt/correction text
- **Workspace Module** (32 learnable slots x 2048 dim): persistent latent state with gated updates
- **Cross-Attention Adapters** (one per decoder layer, zero-initialized): let the decoder condition on the workspace

The cross-attention output projections are initialized to **zero**, so at step 0 the model is **exactly** vanilla Qwen. Training gradually opens the connection.

## How It Works: The Internal Math

**Standard LLMs** have one conditioning channel: the token sequence. At step t:
```
p(y_t | y_<t) = softmax(LMHead(Decoder(y_<t)))
```
New information can only enter by changing the sequence and restarting.

**Duplex-1** adds a second conditioning channel -- the workspace **W**:
```
p(y_t | y_<t, W) = softmax(LMHead(Decoder(y_<t; cross-attn to W)))
```
When a correction arrives, instead of restarting:
- Encode correction: `e = Encoder(correction_tokens)`
- Workspace attends to encoded correction: `W_tilde = CrossAttn(W, e, e)`
- Gated update: `delta = MLP(W_tilde)`, `g = sigmoid(Linear([W; W_tilde]))`, **W_new = W + g * delta**

The decoder's next-token distribution changes because W changed, not because the token sequence was rewritten. That's full-duplex: input updates state while output continues.

## Setup

```bash
pip install -r requirements.txt
python scripts/download_qwen.py
```

## Quick Start

### 1. Generate training data
```bash
python scripts/generate_data.py
```

### 2. Train (Phase 1: basic, Phase 2: with corrections)
```bash
python scripts/train.py --phase 1 --max_steps 10000
python scripts/train.py --phase 2 --max_steps 20000 --resume checkpoints/duplex-1.1-1.7b/step_10000.pt
```

### 3. Evaluate
```bash
python scripts/evaluate.py --duplex_ckpt checkpoints/duplex-1.1-1.7b/final.pt --n_samples 200 --ablation
```

### 4. Demo (side-by-side: Qwen vs Duplex)
```bash
python scripts/demo.py --duplex_ckpt checkpoints/duplex-1.1-1.7b/final.pt
```

## Project Structure

```
duplex/
    config.py              -- Model and training configs
    encoder.py             -- Update Encoder (4-layer bidirectional transformer)
    workspace.py           -- Workspace module (32 slots, gated update)
    adapter.py             -- CrossAttentionAdapter (zero-init output projection)
    duplex_model.py        -- Main model: wraps Qwen + injects adapters
    data/
        tasks.py           -- 7 synthetic task generators
        dataset.py         -- HuggingFace-compatible Dataset
    training/
        trainer.py         -- Training loop (frozen Qwen + train adapters)
    inference/
        generate.py        -- Generation with mid-stream workspace update
scripts/
    download_qwen.py       -- Download Qwen3-1.7B-Base
    generate_data.py       -- Generate training data
    train.py               -- Training CLI
    evaluate.py            -- Evaluation CLI
    demo.py                -- Gradio investor demo
CONCEPT1/                  -- Original toy prototype (proof of concept)
```

## Prior Work (CONCEPT1)

The toy prototype in `CONCEPT1/` validated the core idea at small scale (2M baseline vs 5M workspace model). Key ablation result: disabling the workspace update dropped revision accuracy from 55% to 4.4% and raised contradiction rate from 3.6% to 40.2%. This confirmed the full-duplex mechanism works and justified scaling to Duplex-1.1-1.7B.
