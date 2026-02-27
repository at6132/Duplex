from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 105
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    max_seq_len: int = 512
    dropout: float = 0.1

    # Workspace-specific
    n_workspace_slots: int = 16
    n_encoder_layers: int = 2


@dataclass
class DataConfig:
    max_seq_len: int = 512
    num_train_samples: int = 50000
    num_val_samples: int = 5000
    num_test_samples: int = 5000
    task_types: list[str] = field(default_factory=lambda: [
        "fact_correction",
        "variable_substitution",
        "constraint_revision",
        "arithmetic_correction",
        "key_value_update",
    ])
    data_dir: str = "generated_data"


@dataclass
class TrainingConfig:
    model_type: str = "baseline"  # "baseline" or "workspace"
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 50000
    warmup_steps: int = 2000
    log_every: int = 100
    eval_every: int = 2000
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    phase: int = 1  # 1 = basic, 2 = interrupted
    grad_clip: float = 1.0
    seed: int = 42


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.training.model_type}_phase{self.training.phase}"
