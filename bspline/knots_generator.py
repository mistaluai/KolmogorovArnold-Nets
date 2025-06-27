import torch


def generate_knots(
    low: float,
    high: float,
    k: int,
    G: int,
    in_dim: int,
    out_dim: int,
) -> torch.Tensor:
    """
    Generate a 3D tensor of B-spline control points (knots) for a given input and output dimension.

    This function creates grid points spaced evenly between `low` and `high`, extended by `k` points
    on each side to support B-spline of order `k`. The resulting grid is then broadcasted to shape
    (out_dim, in_dim, G + 2k + 1) to match the dimensions needed for evaluating B-spline basis functions.

    Args:
        low (float): Lower bound of the grid.
        high (float): Upper bound of the grid.
        k (int): Spline order.
        G (int): Number of internal grid intervals.
        in_dim (int): Number of input dimensions (features).
        out_dim (int): Number of output dimensions (units).
        device (torch.device): Device on which to create the tensor.

    Returns:
        torch.Tensor: A tensor of shape (out_dim, in_dim, G + 2k + 1) containing the grid points.
    """
    step = (high - low) / G
    grid = torch.arange(-k, G + k + 1)
    grid = grid * step + low
    grid = grid[None, None, :].expand(out_dim, in_dim, -1).contiguous()
    return grid


if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    grid = generate_knots(
        low=-1.0,
        high=1.0,
        k=3,
        G=10,
        in_dim=3,
        out_dim=1,
    )
    print(grid)