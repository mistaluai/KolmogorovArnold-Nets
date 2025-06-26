import torch
import matplotlib.pyplot as plt


def visualize_bspline_basis(
    x: torch.Tensor,
    bases: torch.Tensor,
    feature_index: int = 0,
    output_index: int = 0
) -> None:
    """
    Visualize B-spline basis function values for a given input feature dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_dim).
        bases (torch.Tensor): B-spline basis tensor of shape
                              (batch_size, out_dim, in_dim, num_bases).
        feature_index (int): Index of the input feature dimension to visualize (default: 0).
        output_index (int): Index of the output dimension to visualize (default: 0).
    """
    # Extract the inputs and basis values for the selected feature and output dim
    x_vals = x[:, feature_index].detach().cpu().numpy()
    basis_vals = bases[:, output_index, feature_index, :].detach().cpu().numpy()

    # Sort by x values for a smoother plot
    sorted_indices = x_vals.argsort()
    x_vals_sorted = x_vals[sorted_indices]
    basis_vals_sorted = basis_vals[sorted_indices]

    # Plot each basis function across all x values
    num_basis_functions = basis_vals.shape[1]
    plt.figure(figsize=(10, 6))
    for i in range(num_basis_functions):
        plt.plot(x_vals_sorted, basis_vals_sorted[:, i], label=f"$B_{i}(x)$")

    plt.title(f"B-spline Basis Functions for Feature {feature_index}, Output {output_index}")
    plt.xlabel("x")
    plt.ylabel("Basis Function Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()