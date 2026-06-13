"""
Source: https://github.com/nanowell/Differential-Transformer-PyTorch

Paper: https://arxiv.org/pdf/2410.05258
Original code: https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .activation import SwiGLU


def build_causal_additive_mask(q_len, k_len, dtype, device, attention_mask=None):
    """Additive attention bias of shape (1 or batch, 1, q_len, k_len).

    0.0 where attention is allowed, -inf otherwise. Query row i has absolute
    position (k_len - q_len + i) and may attend to keys 0..that position (so it
    reduces to a square causal mask when q_len == k_len, and supports cached
    decoding when q_len < k_len). If ``attention_mask`` (batch, k_len) is given,
    padded keys are masked too.
    """
    past = k_len - q_len
    i = torch.arange(q_len, device=device).view(q_len, 1)
    j = torch.arange(k_len, device=device).view(1, k_len)
    bias = torch.zeros(q_len, k_len, dtype=dtype, device=device)
    bias.masked_fill_(j > past + i, float("-inf"))
    bias = bias.view(1, 1, q_len, k_len)
    if attention_mask is not None:
        key_pad = attention_mask.bool().view(-1, 1, 1, k_len)
        bias = bias.masked_fill(~key_pad, float("-inf"))
    return bias


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Applies normalization across the last dimension and scales the output.
    """

    def __init__(self, d, eps=1e-5):
        """
        Args:
            d (int): Dimension of the input features.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d).

        Returns:
            Tensor: Normalized and scaled tensor.
        """
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention Mechanism.
    Replaces the conventional softmax attention with a differential attention.
    Incorporates a causal mask to ensure autoregressive behavior.
    """

    def __init__(self, dim_embd, n_heads, lambda_init):
        """
        Args:
            dim_embd (int): Dimension of the model. Must be divisible by n_heads.
            n_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda.
        """
        super().__init__()
        assert dim_embd % n_heads == 0, "dim_embd must be divisible by n_heads"

        self.n_heads = n_heads
        self.dim_head = dim_embd // n_heads

        # Linear projections for queries, keys, and values
        # Project to 2 * d_head per head for differential attention
        self.W_q = nn.Linear(dim_embd, 2 * self.dim_head * n_heads, bias=False)
        self.W_k = nn.Linear(dim_embd, 2 * self.dim_head * n_heads, bias=False)
        self.W_v = nn.Linear(dim_embd, 2 * self.dim_head * n_heads, bias=False)
        self.W_o = nn.Linear(2 * self.dim_head * n_heads, dim_embd, bias=False)

        # Learnable parameters for lambda reparameterization
        self.lambda_q1 = nn.Parameter(torch.randn(n_heads, self.dim_head))
        self.lambda_k1 = nn.Parameter(torch.randn(n_heads, self.dim_head))
        self.lambda_q2 = nn.Parameter(torch.randn(n_heads, self.dim_head))
        self.lambda_k2 = nn.Parameter(torch.randn(n_heads, self.dim_head))

        self.lambda_init = lambda_init

        # Scale parameter for RMSNorm
        self.rms_scale = nn.Parameter(torch.ones(2 * self.dim_head))
        self.eps = 1e-5  # Epsilon for numerical stability

        # Initialize weights (optional but recommended)
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters for improved training stability.
        """
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.rms_scale, 1.0)

    def forward(self, X, attn_mask=None, past_kv=None, use_cache=False):
        """
        Forward pass for Multi-Head Differential Attention.

        Args:
            X (Tensor): Input tensor of shape (batch, sequence_length, dim_embd).
            attn_mask (Tensor): Additive attention bias broadcastable to
                (batch, n_heads, q_len, k_len), encoding causality and any key
                padding. Built once per forward by the parent model.
            past_kv (tuple): Cached (K, V) from previous steps, or None.
            use_cache (bool): If True, return the updated (K, V) as ``present``.

        Returns:
            (Tensor, tuple|None): the attention output and the (K, V) cache.
        """
        batch, N, dim_embd = X.shape

        # Project inputs to queries, keys, and values
        Q = self.W_q(X)  # Shape: (batch, N, 2 * n_heads * d_head)
        K = self.W_k(X)  # Shape: (batch, N, 2 * n_heads * d_head)
        V = self.W_v(X)  # Shape: (batch, N, 2 * n_heads * d_head)

        # Reshape and permute for multi-head attention
        # New shape: (batch, n_heads, sequence_length, 2 * d_head)
        Q = Q.view(batch, N, self.n_heads, 2 * self.dim_head).transpose(1, 2)
        K = K.view(batch, N, self.n_heads, 2 * self.dim_head).transpose(1, 2)
        V = V.view(batch, N, self.n_heads, 2 * self.dim_head).transpose(1, 2)

        # Prepend cached keys/values for incremental decoding.
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)  # (batch, n_heads, T, 2 * d_head)
            V = torch.cat([past_v, V], dim=2)
        present = (K, V) if use_cache else None

        # Split Q and K into Q1, Q2 and K1, K2
        Q1, Q2 = Q.chunk(2, dim=-1)  # (batch, n_heads, N, d_head)
        K1, K2 = K.chunk(2, dim=-1)  # (batch, n_heads, T, d_head)

        # Compute lambda using reparameterization
        # lambda_val = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init
        # Compute dot products for each head
        # Shape of lambda_val: (n_heads,)
        lambda_q1_dot_k1 = torch.sum(
            self.lambda_q1 * self.lambda_k1, dim=-1
        ).float()  # (n_heads,)
        lambda_q2_dot_k2 = torch.sum(
            self.lambda_q2 * self.lambda_k2, dim=-1
        ).float()  # (n_heads,)
        lambda_val = (
            torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init
        )  # (n_heads,)

        # Expand lambda_val to match attention dimensions
        # Shape: (batch, n_heads, 1, 1)
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # The additive causal (+ padding) mask is built once by the parent model
        # and passed in; fall back to a plain causal mask for standalone use.
        if attn_mask is None:
            attn_mask = build_causal_additive_mask(N, K.size(2), X.dtype, X.device)

        # Compute attention scores
        scaling = 1 / math.sqrt(self.dim_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling  # (batch, n_heads, N, T)
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling  # (batch, n_heads, N, T)

        # Apply the causal (+ padding) mask
        A1 = A1 + attn_mask  # Mask out future / padding positions
        A2 = A2 + attn_mask  # Mask out future / padding positions

        # Apply softmax to get attention weights
        attention1 = F.softmax(A1, dim=-1)  # (batch, n_heads, N, N)
        attention2 = F.softmax(A2, dim=-1)  # (batch, n_heads, N, N)
        attention = attention1 - lambda_val * attention2  # (batch, n_heads, N, N)

        # Apply attention weights to values
        O = torch.matmul(attention, V)  # (batch, n_heads, N, 2 * d_head)  # noqa: E741

        # Normalize each head independently using RMSNorm
        # First, reshape for RMSNorm
        O_reshaped = O.contiguous().view(
            batch * self.n_heads, N, 2 * self.dim_head
        )  # (batch*n_heads, N, 2*d_head)

        # Compute RMSNorm
        rms_norm = torch.sqrt(
            O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )  # (batch*n_heads, N, 1)
        O_normalized = (
            O_reshaped / rms_norm
        ) * self.rms_scale  # (batch*n_heads, N, 2*d_head)

        # Reshape back to (batch, n_heads, N, 2 * d_head)
        O_normalized = O_normalized.view(batch, self.n_heads, N, 2 * self.dim_head)

        # Scale the normalized output
        O_normalized = O_normalized * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        # New shape: (batch, N, n_heads * 2 * d_head)
        O_concat = (
            O_normalized.transpose(1, 2)
            .contiguous()
            .view(batch, N, self.n_heads * 2 * self.dim_head)
        )

        # Final linear projection
        out = self.W_o(O_concat)  # (batch, N, dim_embd)

        return out, present


class DiffTransformerLayer(nn.Module):
    """
    Single Layer of the DiffTransformer Architecture.
    Consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.
    """

    def __init__(self, dim_embd, n_heads, lambda_init):
        """
        Args:
            dim_embd (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda in Differential Attention.
        """
        super().__init__()
        self.norm1 = RMSNorm(dim_embd)
        self.attn = MultiHeadDifferentialAttention(dim_embd, n_heads, lambda_init)
        self.norm2 = RMSNorm(dim_embd)
        self.ff = SwiGLU(dim_embd)

    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False):
        """
        Forward pass for a single transformer layer.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, dim_embd).
            attn_mask (Tensor): Additive attention bias passed to the attention.
            past_kv (tuple): Cached (K, V) for incremental decoding, or None.
            use_cache (bool): If True, return the updated (K, V) cache.

        Returns:
            (Tensor, tuple|None): layer output and the (K, V) cache.
        """
        # Apply Multi-Head Differential Attention with residual connection
        attn_out, present = self.attn(
            self.norm1(x), attn_mask=attn_mask, past_kv=past_kv, use_cache=use_cache
        )
        y = attn_out + x
        # Apply SwiGLU Feed-Forward Network with residual connection
        z = self.ff(self.norm2(y)) + y
        return z, present


class DiffTransformer(nn.Module):
    """
    The DiffTransformer Model incorporating multiple DiffTransformerLayers.
    Suitable for sequence modeling tasks such as language modeling.
    """

    def __init__(self, config):
        """
        Args:
            config (Config): Configuration object.
        """
        super().__init__()

        self.config = config
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        assert self.config.dim_embd % self.config.n_heads == 0, (
            "dim_embd must be divisible by n_heads"
        )

        self.token_emb = nn.Embedding(self.config.vocab_size, self.config.dim_embd)
        self.pos_emb = nn.Embedding(self.config.max_seq_length, self.config.dim_embd)
        self.layers = nn.ModuleList(
            [
                DiffTransformerLayer(
                    dim_embd=self.config.dim_embd,
                    n_heads=self.config.n_heads,
                    lambda_init=0.8
                    - 0.6 * math.exp(-0.3 * (l - 1)),  # Decaying lambda_init
                )
                for l in range(1, self.config.n_layers + 1)  # noqa: E741
            ]
        )
        self.norm = RMSNorm(self.config.dim_embd)
        self.head = nn.Linear(self.config.dim_embd, self.config.vocab_size, bias=False)

        # Initialize weights (optional but recommended)
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters for improved training stability.
        """
        nn.init.xavier_uniform_(self.token_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(
        self,
        x,
        attention_mask=None,
        labels=None,
        output_hidden_states=False,
        past_key_values=None,
        use_cache=False,
    ):
        """
        Forward pass for the DiffTransformer.

        Args:
            x (Tensor): Input token indices of shape (batch, sequence_length).
            attention_mask (Tensor): Optional (batch, N) padding mask.
            labels (Tensor): Optional (batch, N) targets; positions set to -100
                are ignored by the loss. Defaults to the inputs themselves.
            past_key_values (list): Per-layer cached (K, V) for decoding, or None.
            use_cache (bool): If True, return per-layer (K, V) in "past_key_values".

        Returns:
            dict with "logits" and "loss" (and "hidden_states"/"past_key_values").
        """
        batch, N = x.shape
        past_len = past_key_values[0][0].size(2) if past_key_values is not None else 0
        positions = (
            torch.arange(past_len, past_len + N, device=x.device)
            .unsqueeze(0)
            .expand(batch, N)
        )  # (batch, N)
        hidden = self.token_emb(x) + self.pos_emb(positions)  # (batch, N, dim_embd)

        # Build the additive causal (+ padding) mask once, reuse across layers.
        # Keys span the cached prefix (past_len + N); queries are the new tokens.
        attn_mask = build_causal_additive_mask(
            N,
            past_len + N,
            hidden.dtype,
            x.device,
            attention_mask=attention_mask if past_len == 0 else None,
        )

        hidden_states = [hidden] if output_hidden_states else []

        presents = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            hidden, present = layer(
                hidden, attn_mask=attn_mask, past_kv=past, use_cache=use_cache
            )
            presents.append(present)
            if output_hidden_states:
                hidden_states.append(hidden)

        hidden = self.norm(hidden)  # (batch, N, dim_embd)
        logits = self.head(hidden)  # (batch, N, vocab_size)

        outputs = {"logits": logits}
        if use_cache:
            outputs["past_key_values"] = presents

        # Shift logits and labels for next-token prediction. Padding should be
        # supplied via `labels` with -100 (see DataCollator) to be ignored.
        # Skipped during cached decoding (single-token steps have no shift target).
        if not use_cache:
            if labels is None:
                labels = x
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
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

    def generate(self, tokenizer, prompt, max_new_tokens=100, **kwargs):
        """Generate text. Greedy by default; pass do_sample=True with
        temperature/top_k/top_p for sampling. See llamlam.utils.generate."""
        from llamlam.utils import generate

        return generate(
            self, tokenizer, prompt, max_new_tokens=max_new_tokens, **kwargs
        )


# Example usage:

if __name__ == "__main__":
    from .config import Config

    # Define model hyperparameters
    config = Config(
        vocab_size=30522,
        n_heads=12,
        dim_head=64,  # dim_embd = n_heads * dim_head is derived in Config
        n_layers=12,
        max_seq_length=512,
    )

    # Instantiate the model
    model = DiffTransformer(config)

    # Move model to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Example input: batch of token indices
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(
        device
    )  # (batch, N)

    # Forward pass
    logits = model(input_ids)  # (batch, N, vocab_size)
    print(logits.shape)  # Should output: torch.Size([2, 128, 30522])
