from dataclasses import dataclass, field
from typing import Any, Tuple

from dataclass_wizard import YAMLWizard


@dataclass
class BaseConfig(YAMLWizard, key_transform="SNAKE"):
    img_height: int
    img_width: int
    data_path: str
    checkpoint_path: str
    max_files: int
    max_text_len: int

    batch_size: int
    max_epochs: int
    train_size: float
    vocab: str

    def get(self, key: str, default_value: Any | None = None) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return default_value

    def __post_init__(self):
        self.vocab_size: int = len(self.vocab) + 2


@dataclass
class ConfigLatentDiffusion(BaseConfig):
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
    max_seq_len: int
    dataset_txt: str

    use_ema: bool
    num_layers: int
    channels: int
    drop_rate: float
    clip_grad: float
    clip_algorithm: str

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
class ConfigConvNeXt(BaseConfig):
    max_seq_len: int
    gen_model_type: str

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
class ConfigInception(BaseConfig):
    max_seq_len: int
    gen_model_type: str

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
