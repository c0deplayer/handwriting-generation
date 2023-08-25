import string
from dataclasses import dataclass, field

from dataclass_wizard import YAMLWizard


@dataclass
class ConfigLatentDiffusion(YAMLWizard, key_transform="SNAKE"):
    batch_size: int
    max_epochs: int
    img_height: int
    img_width: int
    max_text_len: int
    device: str

    channels: int
    emb_dim: int
    num_heads: int
    num_res_block: int

    interpolation: bool
    data_path: str
    checkpoint_path: str

    vocab: str = field(default=f"_{string.ascii_letters}{string.digits}.?!,'\"-")

    def __post_init__(self):
        self.vocab_size: int = len(self.vocab) + 2


@dataclass
class ConfigDiffusion(YAMLWizard, key_transform="SNAKE"):
    batch_size: int
    max_epochs: int
    img_height: int
    img_width: int
    max_text_len: int
    max_seq_len: int
    max_files: int
    train_size: float
    device: str

    num_layers: int
    channels: int
    drop_rate: int
    clip_grad: float
    clip_algorithm: str

    data_path: str
    dataset_txt: str
    checkpoint_path: str

    vocab: str = field(default=f"_{string.ascii_letters}{string.digits}.?!,'\"-")

    blacklist: tuple[str, ...] = field(
        default=(
            "z00-001",
            "a08-551z",
            "z01-000",
            "z01-000z",
            "z01-010a",
            "z01-010",
            "z01-010b",
            "z01-010c",
        )
    )

    def __post_init__(self):
        self.vocab_size: int = len(self.vocab) + 2


@dataclass
class ConfigRNN(YAMLWizard, key_transform="SNAKE"):
    batch_size: int
    max_epochs: int
    max_text_len: int
    max_seq_len: int
    max_files: int
    train_size: float
    device: str

    input_size: int
    hidden_size: int
    num_window: int
    num_mixture: int
    lstm_clip: float
    mdn_clip: float
    bias: float

    data_path: str
    dataset_txt: str
    checkpoint_path: str

    vocab: str = field(default=f"_{string.ascii_letters}{string.digits}.?!,'\"-")

    blacklist: tuple[str, ...] = field(
        default=(
            "z00-001",
            "a08-551z",
            "z01-000",
            "z01-000z",
            "z01-010a",
            "z01-010",
            "z01-010b",
            "z01-010c",
        )
    )

    def __post_init__(self):
        self.vocab_size: int = len(self.vocab) + 2
