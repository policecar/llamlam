"""
Source: https://github.com/nanowell/Differential-Transformer-PyTorch

Paper: https://arxiv.org/pdf/2410.05258
Original code: https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .activation import SwiGLU
from .generation import generate as _generate
from .model import causal_lm_loss, check_cache_inputs


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
            dim_embd (int): Dimension of the model. Must be divisible by 2 * n_heads.
            n_heads (int): Number of *differential* heads (half a baseline
                Transformer's head count, as in the reference); each is a pair
                of softmax maps over head_dim = dim_embd // n_heads // 2.
            lambda_init (float): Initial value for lambda.

        @decision DEC-001: Reference (unilm multihead_diffattn.py) projection layout:
        Q, K, V and output are all dim_embd -> dim_embd; Q/K hold 2 * n_heads
        components of head_dim, V holds n_heads values of 2 * head_dim.
        @rationale: Projecting to 2 * dim_embd doubled the attention weights
        (8 d^2 vs GPT's 4 d^2); this layout makes the layer parameter-matched.

        @decision DEC-002: One lambda per layer, shared by all heads (reference).
        @rationale: The reference's lambda_q1/k1/q2/k2 are (head_dim,) vectors,
        giving a single scalar lambda per layer; per-head lambdas deviated
        from the paper's formulation.
        """
        super().__init__()
        if dim_embd % (2 * n_heads) != 0:
            raise ValueError(
                f"dim_embd ({dim_embd}) must be divisible by 2 * n_heads "
                f"({2 * n_heads}) for differential attention"
            )

        self.n_heads = n_heads
        self.dim_head = dim_embd // n_heads // 2

        # Linear projections for queries, keys, and values.
        # Viewed per head as 2 * dim_head: Q/K split into (Q1, Q2)/(K1, K2) of
        # dim_head each; V stays whole (2 * dim_head) per head.
        self.W_q = nn.Linear(dim_embd, dim_embd, bias=False)
        self.W_k = nn.Linear(dim_embd, dim_embd, bias=False)
        self.W_v = nn.Linear(dim_embd, dim_embd, bias=False)
        self.W_o = nn.Linear(dim_embd, dim_embd, bias=False)

        # Learnable parameters for lambda reparameterization, shape (dim_head,):
        # one lambda per layer, shared by all heads, as in the reference.
        # std 0.1 as in the paper/reference, so lambda ~= lambda_init at init
        # (std 1 made exp(q . k) range into the thousands for dim_head = 64).
        def _lambda_param():
            return nn.Parameter(torch.zeros(self.dim_head).normal_(0, 0.1))

        self.lambda_q1 = _lambda_param()
        self.lambda_k1 = _lambda_param()
        self.lambda_q2 = _lambda_param()
        self.lambda_k2 = _lambda_param()

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

    def lambda_full(self):
        """
        lambda = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init

        Returns:
            Tensor: scalar (shape ()), shared by all heads of the layer
        """
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        return (
            torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init
        )

    def forward(self, X, past_kv=None, use_cache=False):
        """
        Forward pass for Multi-Head Differential Attention.

        Args:
            X (Tensor): Input tensor of shape (batch, sequence_length, dim_embd),
                only the new positions when past_kv is given.
            past_kv (tuple, optional): cached (K, V), each of shape
                (batch, n_heads, past_length, 2 * d_head), K before the Q1/Q2
                split.
            use_cache (bool): If True, also return the updated (K, V).

        Returns:
            Tensor: Output tensor after applying differential attention, or
            (output, (K, V)) when use_cache is True.
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

        # Prepend cached keys/values: T = past_length + N keys, N queries
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat((past_K, K), dim=2)  # (batch, n_heads, T, 2 * d_head)
            V = torch.cat((past_V, V), dim=2)
        T = K.size(2)
        past_len = T - N

        # Split Q and K into Q1, Q2 and K1, K2
        Q1, Q2 = Q.chunk(2, dim=-1)  # Each of shape: (batch, n_heads, N, d_head)
        K1, K2 = K.chunk(2, dim=-1)  # Each of shape: (batch, n_heads, T, d_head)

        # Compute lambda using reparameterization: one scalar for all heads
        lambda_val = self.lambda_full()

        # ------------------- Causal Mask Implementation ------------------- #
        # Create a causal mask to prevent attention to future tokens.
        # Query i sits at absolute position past_len + i and sees keys
        # 0..past_len + i; without a cache (past_len = 0) this is tril(N, N).
        mask = (
            torch.tril(torch.ones((N, T), device=X.device), diagonal=past_len)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, N, T)
        # Replace 1s with 0.0 and 0s with -inf
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        # -------------------------------------------------------------------- #

        # Compute attention scores
        scaling = 1 / math.sqrt(self.dim_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling  # (batch, n_heads, N, T)
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling  # (batch, n_heads, N, T)

        # Apply the causal mask
        A1 = A1 + mask  # Mask out future positions
        A2 = A2 + mask  # Mask out future positions

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

        if use_cache:
            return out, (K, V)
        return out


class DiffTransformerLayer(nn.Module):
    """
    Single Layer of the DiffTransformer Architecture.
    Consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.
    """

    def __init__(self, dim_embd, n_heads, lambda_init, dropout=0.0, bias=True):
        """
        Args:
            dim_embd (int): Dimension of the model.
            n_heads (int): Number of differential heads (see
                MultiHeadDifferentialAttention), not the baseline head count.
            lambda_init (float): Initial value for lambda in Differential Attention.
            dropout (float): Residual dropout rate (as in GPT's Block).
            bias (bool): Biases in the feed-forward layers (config.bias).
        """
        super().__init__()
        self.norm1 = RMSNorm(dim_embd)
        self.attn = MultiHeadDifferentialAttention(dim_embd, n_heads, lambda_init)
        self.norm2 = RMSNorm(dim_embd)
        self.ff = SwiGLU(dim_embd, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        """
        Forward pass for a single transformer layer.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, dim_embd).
            past_kv, use_cache: see MultiHeadDifferentialAttention.forward.

        Returns:
            Tensor: Output tensor after processing through the layer, or
            (output, (K, V)) when use_cache is True.
        """
        # Apply Multi-Head Differential Attention with residual connection
        attn = self.attn(self.norm1(x), past_kv=past_kv, use_cache=use_cache)
        if use_cache:
            attn, present = attn
        y = self.dropout(attn) + x
        # Apply SwiGLU Feed-Forward Network with residual connection
        z = self.dropout(self.ff(self.norm2(y))) + y
        if use_cache:
            return z, present
        return z


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
        self.loss_fn = nn.CrossEntropyLoss()

        # @decision DEC-003: config.n_heads is the baseline (GPT) head count; the
        # model builds n_heads // 2 differential heads of two softmax maps each.
        # @rationale: Reference ("num_heads set to half of baseline Transformer's
        # num_heads"): each map then has a GPT head's width (config.dim_head),
        # so one config compares both models like-for-like.
        n_heads = self.config.n_heads
        if n_heads < 2 or n_heads % 2 != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be even and >= 2 for DiffTransformer: "
                "it is the baseline head count, and the model builds "
                "n_heads // 2 differential heads of two softmax maps each"
            )
        self.token_emb = nn.Embedding(self.config.vocab_size, self.config.dim_embd)
        self.pos_emb = nn.Embedding(self.config.max_seq_length, self.config.dim_embd)
        self.layers = nn.ModuleList(
            [
                DiffTransformerLayer(
                    dim_embd=self.config.dim_embd,
                    n_heads=n_heads // 2,
                    lambda_init=0.8
                    - 0.6 * math.exp(-0.3 * (l - 1)),  # Decaying lambda_init
                    dropout=self.config.dropout,
                    bias=self.config.bias,
                )
                for l in range(1, self.config.n_layers + 1)  # noqa: E741
            ]
        )
        self.norm = RMSNorm(self.config.dim_embd)
        self.head = nn.Linear(self.config.dim_embd, self.config.vocab_size, bias=False)

        # Initialize weights (optional but recommended)
        self._reset_parameters()

        # @decision DEC-004: config.tie_word_embeddings shares token_emb.weight
        # with head.weight, as in GPTModel; no init change.
        # @rationale: both are (vocab_size, dim_embd) with the same fan-in and
        # fan-out, so xavier_uniform gives them the same distribution; the shared
        # matrix keeps token_emb's init and the head's draw is dropped.
        if self.config.tie_word_embeddings:
            self.head.weight = self.token_emb.weight

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
        output_hidden_states=False,
        past_key_values=None,
        use_cache=False,
    ):
        """
        Forward pass for the DiffTransformer.

        Args:
            x (Tensor): Input tensor of token indices of shape (batch, sequence_length),
                only the new tokens when past_key_values is given.
            attention_mask (Tensor, optional): (batch, sequence_length), 1 = real
                token, 0 = padding. Padded targets are excluded from the loss;
                logits are unaffected. Not allowed in cached mode (no loss).
            past_key_values (list, optional): per-layer (K, V) from a previous call.
            use_cache (bool): return the updated per-layer (K, V).

        Returns:
            dict: "logits" (batch, sequence_length, vocab_size); "loss" unless
            cached (use_cache or past_key_values); "past_key_values" if
            use_cache; optionally "hidden_states".

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
        batch, N = x.shape
        past_len = check_cache_inputs(
            N, past_key_values, len(self.layers), self.config.max_seq_length
        )
        positions = (
            torch.arange(past_len, past_len + N, device=x.device)
            .unsqueeze(0)
            .expand(batch, N)
        )  # (batch, N)
        hidden = self.token_emb(x) + self.pos_emb(positions)  # (batch, N, dim_embd)

        hidden_states = [hidden] if output_hidden_states else []

        presents = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden = layer(hidden, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                hidden, present = hidden
                presents.append(present)
            if output_hidden_states:
                hidden_states.append(hidden)

        hidden = self.norm(hidden)  # (batch, N, dim_embd)
        logits = self.head(hidden)  # (batch, N, vocab_size)

        outputs = {"logits": logits}

        if use_cache:
            outputs["past_key_values"] = presents
        # Calculate loss unless decoding with a cache
        if not cached:
            outputs["loss"] = causal_lm_loss(self.loss_fn, logits, x, attention_mask)

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


# Example usage:

if __name__ == "__main__":
    from .config import Config

    # Define model hyperparameters
    config = Config(
        vocab_size=30522,
        n_heads=12,
        dim_head=64,  # dim_embd = n_heads * dim_head = 768
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
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    print(outputs["logits"].shape)  # Should output: torch.Size([2, 128, 30522])
    print(f"loss: {outputs['loss'].item():.4f}")
