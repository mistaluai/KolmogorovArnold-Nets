import torch
import torch.nn as nn

from bspline.basis_function import calculate_bspline
from bspline.knots_generator import generate_knots


class Spline(nn.Module):
    """
    A Spline activation layer that computes a learnable combination of B-spline basis functions
    for each input feature using the Cox–de Boor formula.

    Args:
        in_dim (int): Number of input features.
        out_dim (int): Number of output units.
        k (int): Order of the B-spline.
        G (int): Number of internal intervals in the spline grid.
        g_low (float): Lower bound of the input domain.
        g_high (float): Upper bound of the input domain.
        device (torch.device): Device to run computations on.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int,
        G: int,
        g_low: float,
        g_high: float,
    ):
        super(Spline, self).__init__()

        self.k = k
        self.activation = calculate_bspline

        # Create and register grid (knot) points
        grid = generate_knots(
            low=g_low,
            high=g_high,
            k=k,
            G=G,
            in_dim=in_dim,
            out_dim=out_dim,
        )
        self.register_buffer("grid", grid)

        # Learnable coefficients for B-spline basis
        self.coeff = nn.Parameter(torch.empty(out_dim, in_dim, G + k))

        self._initialize()

    def _initialize(self):
        """Initialize spline coefficients using Xavier initialization."""
        nn.init.xavier_uniform_(self.coeff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes the spline activation output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim)
        """
        bases = self.activation(x, self.grid, self.k)  # (bsz, out_dim, in_dim, G+k)
        weighted = bases * self.coeff[None, ...]  # Broadcasting coefficients
        output = torch.sum(weighted, dim=-1)  # Sum over basis functions: shape (bsz, out_dim, in_dim)
        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test setup
    batch_size = 4
    in_dim = 3
    out_dim = 2
    k = 3
    G = 5
    g_low = -1.0
    g_high = 1.0

    x = torch.randn(batch_size, in_dim).to(device)
    model = Spline(in_dim, out_dim, k, G, g_low, g_high, device).to(device)
    output = model(x)

    print(f"Input x: {x.shape}")
    print(f"Output: {output.shape}")
    print(output)