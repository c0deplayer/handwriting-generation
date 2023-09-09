from dataclasses import dataclass, field
from typing import Any, Tuple

from dataclass_wizard import YAMLWizard


@dataclass
class BaseConfig(YAMLWizard, key_transform="SNAKE"):
    device: str
    batch_size: int
    max_epochs: int
    max_files: int
    max_text_len: int
    train_size: float

    data_path: str
    checkpoint_path: str

    vocab: str

    def get(self, key: str, default_value: Any = None) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return default_value

    def __post_init__(self):
        self.vocab_size: int = len(self.vocab) + 2


@dataclass
class ConfigLatentDiffusion(BaseConfig):
    img_height: int
    img_width: int

    autoencoder_path: str
    channels: int
    emb_dim: int
    n_heads: int
    res_layers: int
    n_steps: int
    d_cond: int
    beta_start: float
    beta_end: float
    attention_levels: tuple
    channel_multipliers: tuple
    drop_rate: float


@dataclass
class ConfigDiffusion(BaseConfig):
    img_height: int
    img_width: int
    max_seq_len: int

    num_layers: int
    channels: int
    drop_rate: float
    clip_grad: float
    clip_algorithm: str

    dataset_txt: str

    blacklist: Tuple[str, ...] = field(
        default=(
            "z00-001",
            "a08-551z",
            "z01-000",
            "z01-000z",
            "z01-010a",
            "z01-010",
            "z01-010b",
            "z01-010c",
        ),
    )


@dataclass
class ConfigRNN(BaseConfig):
    max_seq_len: int

    input_size: int
    hidden_size: int
    num_window: int
    num_mixture: int
    lstm_clip: float
    mdn_clip: float
    bias: float

    dataset_txt: str

    blacklist: Tuple[str, ...] = field(
        default=(
            "z00-001",
            "a08-551z",
            "z01-000",
            "z01-000z",
            "z01-010a",
            "z01-010",
            "z01-010b",
            "z01-010c",
        ),
    )
