import torch
from bspline.knots_generator import generate_knots
from bspline.visualization import visualize_bspline_basis


def calculate_bspline(
    x: torch.Tensor,
    grid: torch.Tensor,
    k: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute B-spline basis functions of order `k` for each input in `x`
    using the Cox–de Boor recursive formula.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_dim).
        grid (torch.Tensor): Tensor of knot values of shape (out_dim, in_dim, G + 2k + 1),
                             generated using `generate_knots`.
        k (int): Order of the B-spline.
        device (torch.device): Torch device (CPU, CUDA, or MPS).

    Returns:
        torch.Tensor: Tensor of shape (batch_size, out_dim, in_dim, G + k) containing
                      the evaluated basis functions for each input dimension and output channel.
    """
    # Shape (1, out_dim, in_dim, G + 2k + 1)
    grid = grid.unsqueeze(0).to(device=device)

    # Shape (batch_size, 1, in_dim, 1)
    x = x.unsqueeze(1).unsqueeze(-1).to(device=device)

    # Base case: B_i^0(x) = 1 if x in [t_i, t_{i+1}], else 0
    bases = (x >= grid[:, :, :, :-1]) & (x <= grid[:, :, :, 1:])

    # Recursively compute B-spline basis functions
    for j in range(1, k + 1):
        n = grid.size(-1) - (j + 1)

        t_i = grid[:, :, :, :n]
        t_i_plus_j = grid[:, :, :, j:-1]
        t_i_plus_1 = grid[:, :, :, 1:n+1]
        t_i_plus_j_plus_1 = grid[:, :, :, j+1:]

        B_i = bases[:, :, :, :-1]
        B_i_plus_1 = bases[:, :, :, 1:]

        with torch.no_grad():
            denom1 = t_i_plus_j - t_i
            denom2 = t_i_plus_j_plus_1 - t_i_plus_1

        term1 = ((x - t_i) / denom1) * B_i
        term2 = ((t_i_plus_j_plus_1 - x) / denom2) * B_i_plus_1

        bases = term1 + term2

    return bases


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Test example
    k = 3
    G = 5
    x = torch.linspace(-1, 1, steps=200).unsqueeze(1)
    grid = generate_knots(
        low=-1,
        high=1,
        k=k,
        G=G,
        device=device,
        in_dim=x.size(1),
        out_dim=1
    )

    basis_values = calculate_bspline(x, grid, k, device)
    print(basis_values)
    visualize_bspline_basis(x, basis_values)