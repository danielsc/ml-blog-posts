import logging
import os

import torch
from torch import Tensor, nn
from PIL import Image
from torchvision import transforms


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
  

def predict(model: nn.Module, X: Tensor) -> torch.Tensor:
  with torch.no_grad():
    y_prime = model(X) 
    probabilities = nn.functional.softmax(y_prime, dim=1)
    predicted_indices = probabilities.argmax(1)
  return predicted_indices


def init():
  logging.info('Init started')

  global model
  global device

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  logging.info(f'Device: {device}')
  model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pth')
  # Replace previous line with next line and uncomment main to test locally. 
  # model_path = './pytorch_model/model.pth'

  model = torch.load(model_path, map_location=device)
  model.eval()

  logging.info('Init complete')
  pass


def run(mini_batch):
  logging.info(f'Run started: {__file__}, run({mini_batch}')
  predicted_names = []
  transform = transforms.ToTensor()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  logging.info(f'Device: {device}')

  for image_path in mini_batch:
    image = Image.open(image_path)
    tensor = transform(image).to(device)
    predicted_index = predict(model, tensor).item()
    predicted_names.append(labels_map[predicted_index])

  logging.info('Run completed')
  return predicted_names


# if __name__ == '__main__':
#   init()
#   image_paths = [f'sample_request/{filename}' for filename in os.listdir('sample_request')]
#   image_paths.sort()
#   print(run(image_paths))  