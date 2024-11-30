# fmt: off

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:

    # Shared config
    seed: int = 137
    max_seq_length: int = 1024              # max sequence length aka context length

    # Dataset config
    data_path: str = "HuggingFaceTB/smollm-corpus"
    data_name: Optional[str] = "cosmopedia-v2"
    data_files: Optional[str] = "cosmopedia-v2/train-00005-of-00104.parquet"
    # data_path: str = "wikitext"
    # data_name: Optional[str] = "wikitext-103-raw-v1"
    # data_files: Optional[str] = None
    vocab_size: int = 50257                 # number of tokens
    n_data_workers: int = 4

    # Tokenizer config
    pad_to_multiple_of: Optional[int] = None  # when using mixed precision,
                                            # make sure to pad to multiples of 8/16
                                            # mixed_precision == "fp8" uses 16
                                            # mixed_precision != "no" uses 8

    # Model config
    n_layers: int = 12                      # number of layers
    n_heads: int = 12                       # number of heads
    dim_head: int = 64                      # dimensionality of each attention head
    dim_embd: int = field(init=False)       # embedding dimension
    bias: bool = False                      # whether to estimate biased or unbiased std in linear layers, layer norms
                                            # GPT-2 used True; here we default to False which is slightly faster, better
    dropout: float = 0.1                    # dropout rate
    # qkv_bias: bool = True                 # use bias in qkv projection

    # Training config
    n_epochs: int = 3
    eval_steps: int = 200
    learning_rate: float = 6e-4  # [6e-4, 6e-5]
    batch_size: int = 4  # 32
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.01
    no_decay: list[str] = field(default_factory=lambda: ["bias", "LayerNorm.weight"])
    lr_scheduler_type: str = "linear"
    n_warmup_steps: int = 100               # int([0.01, 0.2] * total_steps)
    bfloat16: dict[str, bool] = field(default_factory=lambda: {"enabled": False})
    gradient_clipping: float = 1.0


    def __post_init__(self):
        self.dim_embd = self.n_heads * self.dim_head

# fmt: on
