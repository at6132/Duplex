from dataclasses import dataclass, field

REVISION_MARKERS = {
    "revise_start": "[[REVISE:",
    "revise_end": "]]",
    "insert": "[[INSERT:",
}

SPECIAL_TOKENS = REVISION_MARKERS


@dataclass
class DuplexConfig:
    # Gemma 3 1B-IT architecture
    d_model: int = 1536
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 256
    n_decoder_layers: int = 26
    intermediate_size: int = 6144
    vocab_size: int = 262144
    max_seq_len: int = 8192

    # Workspace (high-level context conditioning)
    n_workspace_slots: int = 32
    workspace_dim: int = 1536

    # Update Encoder
    n_encoder_layers: int = 4
    encoder_dim: int = 1024
    encoder_ff_dim: int = 4096

    # Adapter / Encoder
    adapter_n_heads: int = 16
    adapter_dropout: float = 0.05

    # Model paths & precision
    model_path: str = "google/gemma-3-1b-it"
    quantize_4bit: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 100000
    warmup_steps: int = 3000
    log_every: int = 25
    eval_every: int = 1000
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints/duplex-1.5-4b"
    phase: int = 1
    grad_clip: float = 1.0
    seed: int = 42
    max_seq_len: int = 64

    # DDP (disabled for local single-GPU)
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
    max_response_len: int = 64
    max_correction_len: int = 96
