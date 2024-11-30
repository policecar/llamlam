import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

from llamlam.activation import GELU


class LayerNorm(nn.Module):
    """
    LayerNorm as described in https://arxiv.org/abs/1607.06450
    LayerNorm = ((x - mean) / sqrt(variance + epsilon)) * gamma + beta

    Args:
        ndim: number of dimensions of the input tensor
        bias: whether to estimate a biased or unbiased standard deviation
        eps: small value to prevent division by zero or very small variance
    """

    def __init__(self, ndim, bias, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input):
        return F.layer_norm(
            input,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )


class Attention(nn.Module):
    def __init__(self, dim_embd, n_heads, alpha=0.5):
        super().__init__()

        self.dim_embd = dim_embd
        self.n_heads = n_heads
        self.dim_head = dim_embd // n_heads  # because efficiency
        assert (
            self.dim_head * n_heads == dim_embd
        ), "dim_head should be dim_embd // n_heads because efficiency"

        # Scale factor for dot product attention
        # see Attention is All You Need paper (Vaswani et al., 2017), page 4:
        # "We suspect that for large values of d_k, the dot products grow large in magnitude,
        #  pushing the softmax function into regions where it has extremely small gradients.
        #  To counteract this effect, we scale the dot products by 1 / sqrt(d_k)."
        self.scaling = self.dim_head**-0.5  # not used

        self.qkv_proj = nn.Linear(dim_embd, 3 * dim_embd, bias=False)
        self.out_proj = nn.Linear(dim_embd, dim_embd, bias=False)

        # Initialization for linear layers
        for name, param in self.qkv_proj.named_parameters():
            if "weight" in name:
                init.normal_(param, mean=0, std=alpha * (1 / dim_embd) ** 0.5)
        for name, param in self.out_proj.named_parameters():
            if "weight" in name:
                init.normal_(param, mean=0, std=alpha * (1 / dim_embd) ** 0.5)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(
            batch_size, seq_length, self.n_heads, 3 * self.dim_head
        )  # [B, L, n_heads, 3 * d]
        q, k, v = qkv.chunk(3, dim=-1)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # [B n_heads L d]

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=True, scale=1 / self.dim_head
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_length, self.dim_embd
        )
        output = self.out_proj(attn_output)
        return output


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = Attention(config.dim_embd, config.n_heads)

        self.feedforward = nn.Sequential(
            nn.Linear(config.dim_embd, 4 * config.dim_embd),
            GELU(),
            nn.Linear(4 * config.dim_embd, config.dim_embd),
        )
        for name, param in self.feedforward.named_parameters():
            if "weight" in name:
                init.normal_(param, mean=0, std=0.5 * (1 / config.dim_embd) ** 0.5)
            else:
                init.zeros_(param)

        # Alternative implementation using Conv1D
        # self.feedforward = nn.Sequential(
        #     nn.Conv1D(config.dim_embd, 4 * config.dim_embd),
        #     GELU(),
        #     nn.Conv1D(4 * config.dim_embd, config.dim_embd),
        # )

        self.dropout = nn.Dropout(p=config.dropout)
        self.norm_1 = LayerNorm(config.dim_embd, bias=config.bias, eps=1e-5)
        self.norm_2 = LayerNorm(config.dim_embd, bias=config.bias, eps=1e-5)

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
        self.embed = nn.Embedding(config.vocab_size, config.dim_embd)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_length, config.dim_embd)
        )
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.dim_embd)
        self.head = nn.Linear(config.dim_embd, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.bias = config.bias
        self.dropout = config.dropout

        init.normal_(self.head.weight, mean=0, std=alpha * (1 / config.dim_embd))
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

    def save_pretrained(self, output_dir, tag, optimizer=None):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), output_dir / f"model_{tag}.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), output_dir / f"optimizer_{tag}.pt")

    @classmethod
    def from_pretrained(cls, config, args):
        model = cls(config)
        model.load_state_dict(torch.load(args["model_name_or_path"], weights_only=True))
        model.eval()
        return model

    def generate(self, tokenizer, prompt, max_new_tokens=100):
        """
        Generate text from the model.

        Args:
            tokenizer: Tokenizer instance
            prompt: Text to start generation with
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Decoded text
        """
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.eval()
        self.to(device)

        # Encode prompt
        token_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # TeuxDeux:
        # - top_k
        # - top_p
        # - temperature

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop current context if it exceeds the supported context size
                # E.g., if LLM supports only 5 tokens, and the context size is 10
                # then only the last 5 tokens are used as context
                idx_cond = token_ids[:, -self.config.max_seq_length :]

                # Get the predictions
                with torch.no_grad():
                    logits = self(idx_cond)["logits"]

                # Focus only on the last time step
                # (batch, n_token, vocab_size) becomes (batch, vocab_size)
                logits = logits[:, -1, :]

                # Get the idx of the vocab entry with the highest logits value
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

                # Append sampled index to the running sequence
                token_ids = torch.cat(
                    (token_ids, idx_next), dim=1
                )  # (batch, n_tokens+1)

        # Decode and return the generated text
        return tokenizer.decode(token_ids[0], skip_special_tokens=True)
