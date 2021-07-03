import torch
from torch import nn


class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.sequence = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28*28, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10)
    )

  def forward(self, x: torch.Tensor):
    y_prime = self.sequence(x)
    return y_prime
