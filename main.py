import torch

from KAN_implementation.KAN_model import KAN
from utils.plot import plot
from utils.train_model_adam import train
from utils.visualize_results import visualize_results, plot_against_groundtruth


def f(input):
    x = input[:, 0]
    y = input[:, 1]
    return torch.exp(torch.sin(x) + y).unsqueeze(1)


device = "mps"

model = KAN(
    layer_dimensions=[2, 5, 3, 2, 1],
    k=4,
    G=10,
    G_interval=[-1.0, 1.0],
    device=device,
).to(device)

results = train(model, f, device=device, steps=2000, batch_size=512)
visualize_results(results)
plot_against_groundtruth(model, f, device=device)
plot(model)