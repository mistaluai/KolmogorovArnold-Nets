import torch

from KAN_implementation.KAN_layer import KANLayer
from KAN_implementation.KAN_model import KAN


def l1_regularization(model: KAN):
    layers: list[KANLayer] = model.layers
    l1 = torch.tensor(0.)
    for layer in layers:
        activations = layer.activations
        abs_activations = torch.abs(activations)
        mean = torch.mean(abs_activations)
        sum = torch.sum(mean, dim=0)
        l1 += sum

    return l1

def entropy_regularization(model: KAN):
    reg = torch.tensor(0.)
    eps = 1e-4
    layers: list[KANLayer] = model.layers
    for layer in layers:
        acts = layer.activations
        l1_activations = torch.sum(torch.mean(torch.abs(acts), dim=0))
        activations = (
            torch.mean(torch.abs(l1_activations), dim=0)
            / l1_activations
        )
        entropy = -torch.sum(activations * torch.log(activations + eps))
        reg += entropy

    return reg

def regularization(model: KAN,
                   l1_weight: float=1,
                   entropy_weight: float=1):
    l1 = l1_regularization(model)
    entropy = entropy_regularization(model)
    
    return l1_weight * l1 + entropy_weight * entropy
