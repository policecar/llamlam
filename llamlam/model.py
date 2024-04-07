import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

from activation import GELU


class LMConfig:
    def __init__(self, vocab_size, max_length, n_layer, n_head, n_embd):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd


class Attention(nn.Module):

    def __init__(self, embed_dim, n_heads, alpha=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert (
            self.head_dim * n_heads == embed_dim
        ), "embed_dim must be divisible by n_heads"

        # Scale factor for dot product attention
        # see Attention is All You Need paper (Vaswani et al., 2017), page 4:
        # "We suspect that for large values of d_k, the dot products grow large in magnitude,
        # pushing the softmax function into regions where it has extremely small gradients.
        # To counteract this effect, we scale the dot products by 1 / sqrt(d_k)."
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Initialization for linear layers
        for name, param in self.qkv_proj.named_parameters():
            if "weight" in name:
                init.normal_(param, mean=0, std=alpha * (1 / embed_dim) ** 0.5)
        for name, param in self.out_proj.named_parameters():
            if "weight" in name:
                init.normal_(param, mean=0, std=alpha * (1 / embed_dim) ** 0.5)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(
            batch_size, seq_length, self.n_heads, 3 * self.head_dim
        )  # [B, L, n_heads, 3 * d]
        q, k, v = qkv.chunk(3, dim=-1)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # [B n_heads L d]

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=True, scale=1 / self.head_dim
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_length, self.embed_dim
        )
        output = self.out_proj(attn_output)
        return output


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.attention = Attention(config.n_embd, config.n_head)

        self.feedforward = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )
        for name, param in self.feedforward.named_parameters():
            if "weight" in name:
                init.normal_(param, mean=0, std=0.5 * (1 / config.n_embd) ** 0.5)
            else:
                init.zeros_(param)

        # Alternative implementation using Conv1D
        # self.feedforward = nn.Sequential(
        #     nn.Conv1D(config.n_embd, 4 * config.n_embd),
        #     GELU(),
        #     nn.Conv1D(4 * config.n_embd, config.n_embd),
        # )

        self.dropout = nn.Dropout(p=0.1)
        self.norm_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.norm_2 = nn.LayerNorm(config.n_embd, eps=1e-5)

    def forward(self, x):
        attn = self.attention(self.norm_1(x))
        attn = self.dropout(attn)
        x = x + attn
        mlp = self.feedforward(self.norm_2(x))
        mlp = self.dropout(mlp)
        x = x + mlp
        return x


class GPTModel(nn.Module):

    def __init__(self, config, alpha=0.5):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_length, config.n_embd))
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

        init.normal_(self.head.weight, mean=0, std=alpha * (1 / config.n_embd))
        init.normal_(self.embed.weight, mean=0, std=alpha * 3.3)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        position_ids = torch.arange(
            0, input_ids.size(1), dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        hidden_states = []
        x = self.embed(input_ids) + self.pos_embed[:, : input_ids.size(1), :]
        if output_hidden_states:
            hidden_states.append(x)

        for block in self.blocks:
            x = block(x)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.ln_f(x)
        logits = self.head(x).float()

        outputs = {"logits": logits}
        if input_ids is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            outputs["loss"] = loss

        if output_hidden_states:
            outputs["hidden_states"] = hidden_states

        return outputs
