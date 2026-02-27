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
    vocab_size: int = 151936  # Qwen3 tokenizer vocab; actual may vary slightly
    max_seq_len: int = 4096

    # Workspace
    n_workspace_slots: int = 16
    workspace_dim: int = 2048  # matches d_model

    # Update Encoder (smaller to save VRAM)
    n_encoder_layers: int = 2
    encoder_dim: int = 1024
    encoder_ff_dim: int = 2048

    # Adapter (fewer heads than Qwen to save VRAM)
    adapter_n_heads: int = 8

    # Training
    adapter_dropout: float = 0.05

    # Model paths
    qwen_model_path: str = "models/qwen3-1.7b-base"
    quantize_4bit: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 20000
    warmup_steps: int = 1000
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 2000
    checkpoint_dir: str = "checkpoints/duplex-1-1.7b"
    phase: int = 1
    grad_clip: float = 1.0
    seed: int = 42
    max_seq_len: int = 512


@dataclass
class DataConfig:
    n_train_samples: int = 100000
    n_val_samples: int = 10000
    n_test_samples: int = 5000
    data_dir: str = "generated_data_duplex"
    max_prompt_len: int = 128
    max_response_len: int = 256
    max_correction_len: int = 64
