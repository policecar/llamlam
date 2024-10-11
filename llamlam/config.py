from dataclasses import dataclass

@dataclass
class LMConfig:
    max_length: int = 1024              # max sequence length TODO: rename to context_length
    vocab_size: int = 50257             # number of tokens
    n_layer: int = 12                   # number of layers
    n_head: int = 12                    # number of heads
    head_width: int = 64                # width of each attention head
    n_embd: int = n_head * head_width   # embedding dimension TODO: rename to embed_dim
    # dropout: float = 0.1                # dropout rate
    # qkv_bias: bool = True               # use bias in qkv projection

