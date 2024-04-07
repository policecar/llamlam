import torch
import torch.nn as nn


class Linear(nn.Module):
    """Linear activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GELU(nn.Module):
    """Gaussian Error Linear Unit acc. to https://github.com/hendrycks/GELUs"""

    def __init__(self):
        super().__init__()
        self.const = torch.sqrt(torch.tensor(2.0 / torch.pi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(self.const * (x + 0.044715 * torch.pow(x, 3))))
