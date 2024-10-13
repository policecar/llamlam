# fmt: off

from dataclasses import dataclass, field


@dataclass
class LMConfig:
    max_length: int = 1024                  # max sequence length aka context length
    vocab_size: int = 50257                 # number of tokens
    n_layer: int = 12                       # number of layers
    n_head: int = 12                        # number of heads
    head_width: int = 64                    # width of each attention head
    n_embd: int = field(init=False)         # embedding dimension
    bias: bool = False                      # whether to estimate biased or unbiased std in linear layers, layer norms
                                            # GPT-2 used True; here we default to False which is slightly faster, better
    dropout: float = 0.1                    # dropout rate
    # qkv_bias: bool = True                 # use bias in qkv projection

    def __post_init__(self):
        self.n_embd = self.n_head * self.head_width


@dataclass
class TrainConfig:
    seed: int = 137
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    num_warmup_steps: int = 0
    batch_size: int = 8  # 32
    bfloat16: dict[str, bool] = field(default_factory=lambda: {"enabled": False})
    gradient_clipping: float = 1.0

# fmt: on
