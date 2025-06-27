import torch
import torch.nn as nn

from KAN_implementation.KAN_layer import KANLayer


class KAN(nn.Module):
    """
    Kolmogorov–Arnold Network (KAN) model composed of stacked KANLayers.

    Args:
        layer_dimensions (list[int]): List of integers representing the dimensions of each layer.
        k (int): Number of spline knots per dimension.
        G (int): Number of grid points for the activation function.
        G_interval (list[int]): Interval [g_low, g_high] defining the input range for spline interpolation.
        device (torch.device): Device to run the model on (e.g., torch.device("cuda") or "mps").
    """

    def __init__(self, layer_dimensions: list[int], k: int, G: int, G_interval: list[float], device: torch.device):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = layer_dimensions[:-1]
        out_channels = layer_dimensions[1:]

        for in_dim, out_dim in zip(in_channels, out_channels):
            layer = KANLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                k=k,
                G=G,
                g_low=G_interval[0],
                g_high=G_interval[1],
                device=device,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the stacked KAN layers.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        """
        Custom string representation of the KAN model.
        """
        layer_info = [layer.__repr__() for layer in self.layers]
        return "KAN(\n" + "\n".join(layer_info) + "\n)"


if __name__ == "__main__":
    device = torch.device("mps")
    dimensions = [1, 2, 3, 4]
    k = 3
    G = 5
    G_interval = [-1.0, 1.0]

    model = KAN(dimensions, k, G, G_interval, device)
    print(model)
