
import random
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from neural_network import NeuralNetwork

labels_map = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
  }


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


def visualize_data(dataloader: DataLoader) -> None:
  dataset = dataloader.dataset
  figure = plt.figure(figsize=(8, 8))
  cols = 3
  rows = 3
  for i in range(1, cols * rows + 1):
    sample_idx = random.randint(0, len(dataset))
    (image, label) = dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(image.squeeze(), cmap='gray')
  plt.show()


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
    loss = loss_fn(y, y_prime)

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
  # visualize_data(train_dataloader)

  model = NeuralNetwork().to(device)
  # print(model)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  print('\nFitting:')
  for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}\n-------------------------------')
    fit(device, train_dataloader, model, loss_fn, optimizer)
    
  print('\nEvaluating:')
  (test_loss, test_accuracy) = evaluate(device, test_dataloader, model, loss_fn)
  print(f'Test accuracy: {test_accuracy * 100:>0.1f}%, test loss: {test_loss:>8f}')

  torch.save(model.state_dict(), 'outputs/weights.pth')


def predict(model: nn.Module, X: Tensor) -> torch.Tensor:
  with torch.no_grad():
    y_prime = model(X) 
    probabilities = nn.Softmax(dim=1)(y_prime)
    predicted_indices = probabilities.argmax(1)
  return predicted_indices


def inference_phase(device: str):
  batch_size = 64

  model = NeuralNetwork().to(device)
  model.load_state_dict(torch.load('outputs/weights.pth'))
  model.eval()

  (_, test_dataloader) = get_data(batch_size)
  
  (X_batch, actual_index_batch) = next(iter(test_dataloader))
  X = X_batch[0:3, :, :, :]
  X = X.to(device)
  actual_indices = actual_index_batch[0:3]

  predicted_indices = predict(model, X)

  print('\nPredicting:')
  for (actual_index, predicted_index) in zip(actual_indices, predicted_indices):
    actual_name = labels_map[actual_index.item()]
    predicted_name = labels_map[predicted_index.item()]
    print(f'Actual: {actual_name}, Predicted: {predicted_name}')


def main() -> None:
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  training_phase(device)
  inference_phase(device)


if __name__ == '__main__':
  main()
