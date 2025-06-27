import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from KAN_implementation.KAN_model import KAN
from KAN_implementation.regularization import regularization
from utils.create_dataset import create_dataset
from utils.visualize_results import visualize_results, plot_against_groundtruth


def train(
    model: KAN,
    f,  # target function
    n_var: int = 2,
    ranges=[-1, 1],
    train_num: int = 1000,
    test_num: int = 1000,
    batch_size: int = 128,
    device: torch.device = torch.device("mps"),
    reg_lambda: float = 0.1,
    steps: int = 10000,
    lr: float = 1e-3,
    loss_fn=None,
    loss_fn_eval=None,
    save_path: str = './trained_KAN.pth',
):
    """
    Trains a KAN model on a synthetic function with entropy-based regularization.

    Args:
        model: The KAN model to train.
        f: Target function to approximate.
        n_var: Number of input variables.
        ranges: Input variable ranges.
        train_num: Number of training samples.
        test_num: Number of test samples.
        batch_size: Mini-batch size.
        device: Computation device.
        reg_lambda: Regularization coefficient.
        steps: Number of optimization steps.
        lr: Learning rate.
        loss_fn: Loss function for training.
        loss_fn_eval: Loss function for testing.
        save_path: Path to save best model.
    """

    dataset = create_dataset(f, n_var, ranges, train_num, test_num, device=device)

    if loss_fn is None:
        loss_fn = F.mse_loss
    if loss_fn_eval is None:
        loss_fn_eval = F.mse_loss

    train_loader = DataLoader(
        TensorDataset(dataset["train_input"], dataset["train_label"]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    x_eval, y_eval = dataset["test_input"], dataset["test_label"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    results = {
        "train_loss": [],
        "test_loss": [],
        "regularization": [],
        "best_test_loss": [],
    }

    best_test_loss = float("inf")
    pbar = tqdm(range(steps), desc="KAN Training", ncols=200)

    data_iter = iter(train_loader)

    for step in pbar:
        # Recycle the DataLoader
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        model.train()
        pred = model(x)
        train_loss = loss_fn(pred, y)

        ent_l1_reg = regularization(model, device=device)
        loss = train_loss + reg_lambda * ent_l1_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_pred = model(x_eval)
            test_loss = loss_fn_eval(test_pred, y_eval)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_path)

        if step % 10 == 0:
            pbar.set_description(
                f"Step {step} | Train Loss: {train_loss.item():.2e} | Test Loss: {test_loss.item():.2e} | Reg: {ent_l1_reg.item():.2e}"
            )

        results["train_loss"].append(train_loss.item())
        results["test_loss"].append(test_loss.item())
        results["best_test_loss"].append(best_test_loss.item())
        results["regularization"].append(ent_l1_reg.item())
    model.load_state_dict(torch.load(save_path))
    return results

if __name__ == "__main__":
    def f(input):
        x = input[:, 0]
        y = input[:, 1]
        return torch.exp(torch.sin(torch.pi * x) + y ** 2).unsqueeze(1)


    device = "mps"

    model = KAN(
        layer_dimensions=[2, 32, 64, 32, 1],
        k=3,
        G=5,
        G_interval=[-1.0, 1.0],
        device=device,
    ).to(device)

    results = train(model, f, device=device, steps=5000, batch_size=256)
    visualize_results(results)
    plot_against_groundtruth(model, f, device=device)
