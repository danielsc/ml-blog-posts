import shutil
from typing import Tuple

import mlflow
import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model_wrapper import ModelWrapper
from neural_network import NeuralNetwork


def get_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
  training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
  )

  test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
  )

  train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
  test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

  return (train_dataloader, test_dataloader)


def fit_one_batch(X: torch.Tensor, y: torch.Tensor, model: NeuralNetwork, 
loss_fn: CrossEntropyLoss, optimizer: Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
  y_prime = model(X)
  loss = loss_fn(y_prime, y)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return (y_prime, loss)


def fit(device: str, dataloader: DataLoader, model: nn.Module, loss_fn: CrossEntropyLoss, 
optimizer: Optimizer) -> None:
  batch_count = len(dataloader)
  loss_sum = 0
  correct_item_count = 0
  current_item_count = 0
  print_every = 100

  for batch_index, (X, y) in enumerate(dataloader):
    X = X.to(device)
    y = y.to(device)
    
    (y_prime, loss) = fit_one_batch(X, y, model, loss_fn, optimizer)

    correct_item_count += (y_prime.argmax(1) == y).type(torch.int8).sum().item()
    batch_loss = loss.item()
    loss_sum += batch_loss
    current_item_count += len(X)

    if ((batch_index + 1) % print_every == 0) or ((batch_index + 1) == batch_count):
      batch_accuracy = correct_item_count / current_item_count * 100
      print(f'[Batch {batch_index + 1:>3d} - {current_item_count:>5d} items] accuracy: {batch_accuracy:>0.1f}%, loss: {batch_loss:>7f}')


def evaluate_one_batch(X: torch.tensor, y: torch.tensor, model: NeuralNetwork, 
loss_fn: CrossEntropyLoss) -> Tuple[torch.Tensor, torch.Tensor]:
  with torch.no_grad():
    y_prime = model(X)
    loss = loss_fn(y_prime, y)

  return (y_prime, loss)


def evaluate(device: str, dataloader: DataLoader, model: nn.Module, 
loss_fn: CrossEntropyLoss) -> Tuple[float, float]:
  batch_count = len(dataloader)
  loss_sum = 0
  correct_item_count = 0
  item_count = 0

  model.eval()

  with torch.no_grad():
    for (X, y) in dataloader:
      X = X.to(device)
      y = y.to(device)

      (y_prime, loss) = evaluate_one_batch(X, y, model, loss_fn)

      correct_item_count += (y_prime.argmax(1) == y).type(torch.int8).sum().item()
      loss_sum += loss.item()
      item_count += len(X)

    average_loss = loss_sum / batch_count
    accuracy = correct_item_count / item_count
    return (average_loss, accuracy)


def training_phase(device: str):
  learning_rate = 0.1
  batch_size = 64
  epochs = 2

  (train_dataloader, test_dataloader) = get_data(batch_size)

  model = NeuralNetwork().to(device)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  print('\nFitting:')
  for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}\n-------------------------------')
    fit(device, train_dataloader, model, loss_fn, optimizer)
    
  print('\nEvaluating:')
  (test_loss, test_accuracy) = evaluate(device, test_dataloader, model, loss_fn)
  print(f'Test accuracy: {test_accuracy * 100:>0.1f}%, test loss: {test_loss:>8f}')

  try:
    shutil.rmtree('mlflow-model')
  except:
    pass
  mlflow.pytorch.save_model(model, 'mlflow-model', code_paths=['src/neural_network.py'])


def save_pyfunc_model():
  mlflow_pyfunc_model_path = 'mlflow-pyfunc-model'
  artifacts = {
    'pytorch-model': 'mlflow-model'
  }
  try:
    shutil.rmtree(mlflow_pyfunc_model_path)
  except:
    pass
  model = ModelWrapper()
  mlflow.pyfunc.save_model(mlflow_pyfunc_model_path, python_model=model,
    artifacts=artifacts, code_path=['src/model_wrapper.py'], conda_env='mlflow-model/conda.yaml')


def main() -> None:
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  training_phase(device)
  save_pyfunc_model()


if __name__ == '__main__':
  main()
