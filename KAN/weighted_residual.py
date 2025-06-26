import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedResidual(nn.Module):
    """
    Weighted residual layer used in KAN to combine learned spline activations with
    nonlinear residuals (e.g., SiLU).

    Computes:
        ϕ(x) = w_b * SiLU(x) + w_s * spline(x)

    Args:
        in_dim (int): Number of input features.
        out_dim (int): Number of output features.
        residual_variance (float): Stddev used to initialize residual weights w_b.
    """

    def __init__(self, in_dim: int, out_dim: int, residual_variance: float = 0.1, device="mps"):
        super().__init__()
        self.func = F.silu
        self.wb = nn.Parameter(torch.Tensor(out_dim, in_dim)).to(device)  # residual weights
        self.ws = nn.Parameter(torch.Tensor(out_dim, in_dim)).to(device)  # spline weights
        self._initialize(residual_variance)

    def _initialize(self, residual_variance: float = 0.1) -> None:
        nn.init.normal_(self.wb, mean=0.0, std=residual_variance)
        nn.init.ones_(self.ws)

    def forward(self, x: torch.Tensor, sx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the weighted residual layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).
            sx (torch.Tensor): Spline activations of shape (batch_size, out_dim, in_dim).

        Returns:
            torch.Tensor: Combined activation of shape (batch_size, out_dim, in_dim).
        """
        residuals = self.wb * self.func(x[:, None, :])  # shape: (bsz, out_dim, in_dim)
        post_acts = self.ws * sx                        # shape: (bsz, out_dim, in_dim)
        return residuals + post_acts


if __name__ == "__main__":
    # Test code
    batch_size = 4
    in_dim = 3
    out_dim = 1

    x = torch.randn(batch_size, in_dim)  # random inputs
    sx = torch.randn(batch_size, out_dim, in_dim)  # simulated spline activations

    layer = WeightedResidual(in_dim, out_dim)
    output = layer(x, sx)

    print("Input x:", x)
    print("Spline activations sx:", sx)
    print("Output:", output)