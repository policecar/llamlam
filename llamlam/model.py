import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

from llamlam.activation import GELU
from llamlam.generation import generate as _generate


def check_cache_inputs(n_new, past_key_values, n_layers, max_seq_length):
    """
    Validate a (possibly cached) forward call; return the cache length.

    Shared by GPTModel and DiffTransformer. Learned absolute position
    embeddings only exist for positions 0 .. max_seq_length - 1, so a cache
    plus new tokens longer than that has no valid positions.

    Raises:
        ValueError: on a cache with the wrong number of layers, or a total
            length (cache + new) above max_seq_length.
    """
    past_len = 0
    if past_key_values is not None:
        if len(past_key_values) != n_layers:
            raise ValueError(
                f"past_key_values has {len(past_key_values)} layers, "
                f"the model has {n_layers}"
            )
        past_len = past_key_values[0][0].size(2)
    if past_len + n_new > max_seq_length:
        raise ValueError(
            f"sequence length {past_len + n_new} (cache {past_len} + new {n_new}) "
            f"exceeds max_seq_length {max_seq_length}"
        )
    return past_len


def causal_lm_loss(loss_fn, logits, input_ids, attention_mask=None):
    """
    Next-token prediction loss shared by GPTModel and DiffTransformer.

    @decision DEC-001: Padding is excluded from the loss by setting masked target
    labels to `loss_fn.ignore_index`; logits are never touched.
    @rationale: Attention is causal, so right-padding cannot influence real
    positions; only the loss average was polluted. Masking labels keeps the
    `attention_mask=None` path bit-identical and lets both models share one
    contract (1 = real token, 0 = padding; a target is dropped iff its own mask
    entry is 0).

    Args:
        loss_fn: nn.CrossEntropyLoss (its ignore_index marks dropped targets)
        logits: [B, L, V]
        input_ids: [B, L]
        attention_mask: optional [B, L], 1 = real token, 0 = padding

    Raises:
        ValueError: if attention_mask has the wrong shape, or masks every target.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        target_is_real = attention_mask[..., 1:].bool()
        if not target_is_real.any():
            raise ValueError(
                "attention_mask masks every target position; the loss would be NaN"
            )
        shift_labels = shift_labels.masked_fill(~target_is_real, loss_fn.ignore_index)

    return loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


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


class Context(nn.Module):
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

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        """
        Args:
            x: [B, L, dim_embd], the new positions only when past_kv is given
            mask: passed to SDPA alongside is_causal=True (no-cache path only)
            past_kv: optional cached (K, V), each [B, n_heads, L_past, dim_head]
            use_cache: if True, also return the updated (K, V)

        Returns:
            output [B, L, dim_embd], or (output, (K, V)) when use_cache is True.

        @decision DEC-001: Without past_kv the attention call is unchanged
        (is_causal=True, the fused causal path); only a call with past_kv uses
        an explicit [L, L_past + L] causal mask.
        @rationale: SDPA's is_causal aligns the mask top-left, which is wrong
        once keys outnumber queries; keeping the old call elsewhere keeps the
        training forward and the cache prefill bit-identical to before.
        """
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(
            batch_size, seq_length, self.n_heads, 3 * self.dim_head
        )  # [B, L, n_heads, 3 * d]
        q, k, v = qkv.chunk(3, dim=-1)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # [B n_heads L d]

        if past_kv is None:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=True, scale=1 / self.dim_head
            )
        else:
            if mask is not None:
                raise ValueError("mask is not supported together with past_kv")
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)  # [B n_heads L_past+L d]
            v = torch.cat((past_v, v), dim=2)
            # Query i sits at absolute position L_past + i: it sees keys 0..L_past+i
            past_len = k.size(2) - seq_length
            cross_mask = torch.ones(
                seq_length, k.size(2), dtype=torch.bool, device=x.device
            ).tril(diagonal=past_len)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=cross_mask, scale=1 / self.dim_head
            )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_length, self.dim_embd
        )
        output = self.out_proj(attn_output)
        if use_cache:
            return output, (k, v)
        return output


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.context = Context(config.dim_embd, config.n_heads)

        self.feedforward = nn.Sequential(
            nn.Linear(config.dim_embd, 4 * config.dim_embd, bias=config.bias),
            GELU(),
            nn.Linear(4 * config.dim_embd, config.dim_embd, bias=config.bias),
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

    def forward(self, x, past_kv=None, use_cache=False):
        """Returns x, or (x, (K, V)) when use_cache is True (see Context)."""
        attn = self.context(self.norm_1(x), past_kv=past_kv, use_cache=use_cache)
        if use_cache:
            attn, present = attn
        attn = self.dropout(attn)
        x = x + attn
        mlp = self.feedforward(self.norm_2(x))
        mlp = self.dropout(mlp)
        x = x + mlp
        if use_cache:
            return x, present
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
        self.ln_f = nn.LayerNorm(config.dim_embd, bias=config.bias)
        self.head = nn.Linear(config.dim_embd, config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.bias = config.bias
        self.dropout = config.dropout

        init.normal_(self.head.weight, mean=0, std=alpha * (1 / config.dim_embd))
        init.normal_(self.embed.weight, mean=0, std=alpha * 3.3)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        past_key_values=None,
        use_cache=False,
    ):
        """
        Args:
            input_ids: [B, L] token ids (only the new tokens when cached)
            attention_mask: optional [B, L], 1 = real token; excludes padded
                targets from the loss (no loss, hence not allowed, when cached)
            output_hidden_states: also return per-layer hidden states
            past_key_values: optional per-layer (K, V) from a previous call
            use_cache: return the updated per-layer (K, V) as "past_key_values"

        Returns:
            dict with "logits"; "loss" unless cached (use_cache or
            past_key_values); "past_key_values" if use_cache; "hidden_states".

        Raises:
            ValueError: if cache + new length exceeds max_seq_length, the cache
                has the wrong number of layers, or attention_mask is passed in
                cached mode.
        """
        cached = use_cache or past_key_values is not None
        if cached and attention_mask is not None:
            raise ValueError(
                "attention_mask only masks the loss, which cached mode does not compute"
            )
        seq_length = input_ids.size(1)
        past_len = check_cache_inputs(
            seq_length,
            past_key_values,
            len(self.blocks),
            self.config.max_seq_length,
        )

        hidden_states = []
        x = (
            self.embed(input_ids)
            + self.pos_embed[:, past_len : past_len + seq_length, :]
        )
        if output_hidden_states:
            hidden_states.append(x)

        presents = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                x, present = x
                presents.append(present)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.ln_f(x)
        logits = self.head(x).float()
        outputs = {"logits": logits}

        if use_cache:
            outputs["past_key_values"] = presents
        if not cached:
            outputs["loss"] = causal_lm_loss(
                self.loss_fn, logits, input_ids, attention_mask
            )

        if output_hidden_states:
            outputs["hidden_states"] = hidden_states

        return outputs

    def save_pretrained(self, output_dir, tag, optimizer=None):
        output_dir = Path(output_dir)
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

    def generate(self, tokenizer, prompt, max_new_tokens=100, **kwargs):
        """
        Generate text from the model, greedy by default; see
        llamlam.generation.generate for do_sample / temperature / top_k / top_p.
        """
        return _generate(self, tokenizer, prompt, max_new_tokens, **kwargs)
