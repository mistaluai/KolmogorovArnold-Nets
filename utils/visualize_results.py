import matplotlib.pyplot as plt

def visualize_results(results: dict):
    """
    Plot training metrics from the KAN training loop.

    Args:
        results (dict): Dictionary containing 'train_loss', 'test_loss',
                        'best_test_loss', and 'regularization' as keys.
    """
    steps = range(len(results["train_loss"]))

    plt.figure(figsize=(12, 8))

    # --- Losses ---
    plt.subplot(2, 1, 1)
    plt.plot(steps, results["train_loss"], label="Train Loss", color="blue")
    plt.plot(steps, results["test_loss"], label="Test Loss", color="orange")
    plt.plot(steps, results["best_test_loss"], label="Best Test Loss", color="green", linestyle='--')
    plt.title("Loss Curves")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # --- Regularization ---
    plt.subplot(2, 1, 2)
    plt.plot(steps, results["regularization"], label="Entropy Regularization", color="red")
    plt.title("Entropy Regularization Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Regularization Term")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_against_groundtruth(
        model,
        f,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        resolution=100,
        device=torch.device("cpu")
):
    """
    Plot the ground truth function vs model prediction for 2D inputs.

    Args:
        model (nn.Module): Trained KAN model.
        f (callable): Ground truth function mapping torch.Tensor -> torch.Tensor.
        x_range (tuple): Range for x-axis input values.
        y_range (tuple): Range for y-axis input values.
        resolution (int): Number of grid points per axis.
        device (torch.device): Device to run model on.
    """
    assert model is not None, "Model cannot be None"
    assert callable(f), "Function f must be callable"

    # Create mesh grid
    x = torch.linspace(x_range[0], x_range[1], resolution)
    y = torch.linspace(y_range[0], y_range[1], resolution)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    inputs = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    # Compute outputs
    with torch.no_grad():
        true_vals = f(inputs).cpu().numpy().reshape(resolution, resolution)
        pred_vals = model(inputs).cpu().numpy().reshape(resolution, resolution)

    # Plot ground truth
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), true_vals, levels=50, cmap="viridis")
    plt.colorbar()
    plt.title("Ground Truth")

    # Plot model predictions
    plt.subplot(1, 2, 2)
    plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), pred_vals, levels=50, cmap="viridis")
    plt.colorbar()
    plt.title("Model Prediction")

    plt.suptitle("KAN vs Ground Truth", fontsize=16)
    plt.tight_layout()
    plt.show()