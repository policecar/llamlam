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
    pad_token_id: Optional[int] = None      # padding token id for loss masking

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
        self._validate()

    def _validate(self):
        """Validate configuration parameters to catch errors early."""
        # Positive integer constraints
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.dim_head <= 0:
            raise ValueError(f"dim_head must be positive, got {self.dim_head}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        if self.n_warmup_steps < 0:
            raise ValueError(f"n_warmup_steps must be non-negative, got {self.n_warmup_steps}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if self.eval_steps <= 0:
            raise ValueError(f"eval_steps must be positive, got {self.eval_steps}")
        if self.n_data_workers < 0:
            raise ValueError(f"n_data_workers must be non-negative, got {self.n_data_workers}")

        # Range constraints for floats
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.gradient_clipping < 0:
            raise ValueError(f"gradient_clipping must be non-negative, got {self.gradient_clipping}")

        # Logical constraints
        if self.dim_embd % self.n_heads != 0:
            raise ValueError(f"dim_embd ({self.dim_embd}) must be divisible by n_heads ({self.n_heads})")

        # Scheduler type validation
        valid_schedulers = ["linear", "cosine", "constant", "polynomial"]
        if self.lr_scheduler_type not in valid_schedulers:
            raise ValueError(f"lr_scheduler_type must be one of {valid_schedulers}, got '{self.lr_scheduler_type}'")

        # Optional constraints
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of <= 0:
            raise ValueError(f"pad_to_multiple_of must be positive if specified, got {self.pad_to_multiple_of}")


# fmt: on
