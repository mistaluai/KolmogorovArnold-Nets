import torch
import torch.nn as nn

from KAN_implementation.spline_layer import Spline
from KAN_implementation.weighted_residual import WeightedResidual


class KANLayer(nn.Module):
    """
    KANLayer implements one layer of a Kolmogorov-Arnold Network (KAN).

    It consists of:
    - A learnable B-spline activation function (univariate, per-dimension).
    - A residual weighted connection to learn phi(x) = spline(x) + silu(Wx).

    Args:
        in_dim (int): Number of input features.
        out_dim (int): Number of output units.
        k (int): Spline order.
        G (int): Number of internal grid intervals.
        g_low (float): Lower bound of spline support.
        g_high (float): Upper bound of spline support.
        device (torch.device): Device to run on (CPU, CUDA, or MPS).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int,
        G: int,
        g_low: float,
        g_high: float,
        device: torch.device,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.G = G
        self.device = device

        self.spline = Spline(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            G=G,
            g_low=g_low,
            g_high=g_high,
            device=device,
        )

        self.weighted_residual = WeightedResidual(
            in_dim=in_dim,
            out_dim=out_dim,
        )
        self.input_x = None
        self.activations = None
    def cache(self, x , activations):
        self.input_x = x
        self.activations = activations

    def __repr__(self):
        return f"KANLayer(in_dim={self.in_dim}, out_dim={self.out_dim}, k={self.k}, G={self.G})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the KAN layer.

        Args:
            x (torch.Tensor): Input of shape (batch_size, in_dim)

        Returns:
            torch.Tensor: Output of shape (batch_size, out_dim)
        """
        spline = self.spline(x)  # Shape: (batch_size, out_dim, in_dim)
        phi = self.weighted_residual(x, spline)  # Same shape
        self.cache(x, phi)
        out = torch.sum(phi, dim=-1)  # Sum over in_dim → (batch_size, out_dim)
        return out

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    layer = KANLayer(
        in_dim=5,
        out_dim=2,
        k=3,
        G=10,
        g_low=-1.0,
        g_high=1.0,
        device=device,
    )

    x = torch.randn(4, 5).to(device)
    y = layer(x)

    print("Input:", x.shape)        # (4, 5)
    print("Output:", y.shape)       # (4, 2)
