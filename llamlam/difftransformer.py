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

    def forward(self, X):
        """
        Forward pass for Multi-Head Differential Attention.

        Args:
            X (Tensor): Input tensor of shape (batch, sequence_length, dim_embd).

        Returns:
            Tensor: Output tensor after applying differential attention.
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

        # Split Q and K into Q1, Q2 and K1, K2
        Q1, Q2 = Q.chunk(2, dim=-1)  # Each of shape: (batch, n_heads, N, d_head)
        K1, K2 = K.chunk(2, dim=-1)  # Each of shape: (batch, n_heads, N, d_head)

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

        # ------------------- Causal Mask Implementation ------------------- #
        # Create a causal mask to prevent attention to future tokens
        # Shape of mask: (1, 1, N, N)
        mask = (
            torch.tril(torch.ones((N, N), device=X.device)).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, N, N)
        # Replace 1s with 0.0 and 0s with -inf
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        # -------------------------------------------------------------------- #

        # Compute attention scores
        scaling = 1 / math.sqrt(self.dim_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling  # (batch, n_heads, N, N)
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling  # (batch, n_heads, N, N)

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

        return out


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

    def forward(self, x):
        """
        Forward pass for a single transformer layer.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, dim_embd).

        Returns:
            Tensor: Output tensor after processing through the layer.
        """
        # Apply Multi-Head Differential Attention with residual connection
        y = self.attn(self.norm1(x)) + x
        # Apply SwiGLU Feed-Forward Network with residual connection
        z = self.ff(self.norm2(y)) + y
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

        assert (
            self.config.dim_embd % self.config.n_heads == 0
        ), "dim_embd must be divisible by n_heads"

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

    def forward(self, x, attention_mask=None, output_hidden_states=False):
        """
        Forward pass for the DiffTransformer.

        Args:
            x (Tensor): Input tensor of token indices of shape (batch, sequence_length).

        Returns:
            Tensor: Logits for each token in the vocabulary of shape (batch, sequence_length, vocab_size).
        """
        batch, N = x.shape
        positions = (
            torch.arange(N, device=x.device).unsqueeze(0).expand(batch, N)
        )  # (batch, N)
        hidden = self.token_emb(x) + self.pos_emb(positions)  # (batch, N, dim_embd)

        hidden_states = [hidden] if output_hidden_states else []

        for layer in self.layers:
            hidden = layer(hidden)
            if output_hidden_states:
                hidden_states.append(hidden)

        hidden = self.norm(hidden)  # (batch, N, dim_embd)
        logits = self.head(hidden)  # (batch, N, vocab_size)

        outputs = {"logits": logits}

        # Calculate loss if input tokens provided
        if x is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = x[..., 1:].contiguous()

            # Calculate cross entropy loss
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


# Example usage:

if __name__ == "__main__":
    from .config import Config

    # Define model hyperparameters
    config = Config(
        vocab_size=30522,
        dim_embd=768,
        n_heads=12,
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
