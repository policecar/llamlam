# fmt: off

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:

    # Shared config
    seed: int = 137
    max_length: int = 1024                  # max sequence length aka context length

    # Dataset config
    data_path: str = "HuggingFaceTB/smollm-corpus"
    data_name: Optional[str] = "cosmopedia-v2"
    data_files: Optional[str] = "cosmopedia-v2/train-00005-of-00104.parquet"
    # data_path: str = "wikitext"
    # data_name: Optional[str] = "wikitext-103-raw-v1"
    # data_files: Optional[str] = None
    vocab_size: int = 50257                 # number of tokens
    num_data_workers: int = 4

    # Tokenizer config
    pad_to_multiple_of: Optional[int] = None  # when using mixed precision,
                                            # make sure to pad to multiples of 8/16
                                            # mixed_precision == "fp8" uses 16
                                            # mixed_precision != "no" uses 8

    # Model config
    n_layer: int = 12                       # number of layers
    n_head: int = 12                        # number of heads
    head_width: int = 32  # 64              # width of each attention head
    n_embd: int = field(init=False)         # embedding dimension
    bias: bool = False                      # whether to estimate biased or unbiased std in linear layers, layer norms
                                            # GPT-2 used True; here we default to False which is slightly faster, better
    dropout: float = 0.1                    # dropout rate
    # qkv_bias: bool = True                 # use bias in qkv projection

    # Training config
    num_epochs: int = 3
    eval_steps: int = 200
    learning_rate: float = 1e-4  # 5e-5
    weight_decay: float = 0.01
    no_decay: list[str] = field(default_factory=lambda: ["bias", "LayerNorm.weight"])
    lr_scheduler_type: str = "linear"
    num_warmup_steps: int = 100             # int([0.01, 0.2] * total_steps)
    batch_size: int = 8  # 32
    bfloat16: dict[str, bool] = field(default_factory=lambda: {"enabled": False})
    gradient_clipping: float = 1.0
    gradient_accumulation_steps: int = 1


    def __post_init__(self):
        self.n_embd = self.n_head * self.head_width

# fmt: on
