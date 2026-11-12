import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """Linear activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GELU(nn.Module):
    """Gaussian Error Linear Unit acc. to https://github.com/hendrycks/GELUs"""

    def __init__(self):
        super().__init__()
        self.const = math.sqrt(2.0 / math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x * (1 + torch.tanh(self.const * (x + 0.044715 * torch.pow(x, 3.0))))
        )


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    Combines the Swish activation with Gated Linear Units.

    Taken from https://github.com/nanowell/Differential-Transformer-PyTorch
    """

    def __init__(self, d_model, bias=True):
        """
        Args:
            d_model (int): Dimension of the input features.
            bias (bool): Whether the three linear layers have biases.
        """
        super().__init__()
        # Hidden size 8/3 * d_model (Diff Transformer paper, after LLaMA): three
        # d_model x hidden matrices then hold ~8 * d_model^2 weights, the same as
        # a GELU MLP with hidden 4 * d_model.
        hidden = int(8 * d_model / 3)
        # Intermediate projection layers
        # Typically, SwiGLU splits the computation into two parts
        self.WG = nn.Linear(d_model, hidden, bias=bias)
        self.W1 = nn.Linear(d_model, hidden, bias=bias)
        self.W2 = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x):
        """
        Forward pass for SwiGLU.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying SwiGLU.
        """
        # Apply the gates
        g = F.silu(self.WG(x))  # Activation part
        z = self.W1(x)  # Linear part
        # Element-wise multiplication and projection
        return self.W2(g * z)
