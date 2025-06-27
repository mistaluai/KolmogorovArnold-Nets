import torch

from KAN_implementation.KAN_model import KAN
from utils.train_model_adam import train
from utils.visualize_results import visualize_results, plot_against_groundtruth


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