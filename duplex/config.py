from dataclasses import dataclass, field


@dataclass
class DuplexConfig:
    # Qwen3-1.7B-Base architecture (must match)
    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    n_decoder_layers: int = 28
    intermediate_size: int = 6144
    vocab_size: int = 151936
    max_seq_len: int = 4096

    # Workspace — more slots = richer latent memory
    n_workspace_slots: int = 32
    workspace_dim: int = 2048

    # Update Encoder
    n_encoder_layers: int = 4
    encoder_dim: int = 1024
    encoder_ff_dim: int = 4096

    # Adapter
    adapter_n_heads: int = 16

    # Training
    adapter_dropout: float = 0.05

    # Model paths & precision
    qwen_model_path: str = "models/qwen3-1.7b-base"
    quantize_4bit: bool = False  # H200 has 282 GB VRAM — no quantization needed


@dataclass
class TrainingConfig:
    # H200-optimized defaults: batch 32 per GPU × 2 GPUs × grad_accum 2 = 128 effective
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4      # scaled up for larger effective batch
    weight_decay: float = 0.01
    max_steps: int = 100000
    warmup_steps: int = 5000
    log_every: int = 25
    eval_every: int = 1000
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints/duplex-1-1.7b"
    phase: int = 1
    grad_clip: float = 1.0
    seed: int = 42
    max_seq_len: int = 512

    # DDP (set automatically by train.py)
    use_ddp: bool = False
    world_size: int = 1
    local_rank: int = 0


@dataclass
class DataConfig:
    n_train_samples: int = 500000
    n_val_samples: int = 20000
    n_test_samples: int = 10000
    data_dir: str = "generated_data_duplex"
    max_prompt_len: int = 128
    max_response_len: int = 256
    max_correction_len: int = 64
