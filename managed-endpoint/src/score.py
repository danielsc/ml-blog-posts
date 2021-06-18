import json
import logging
import os
from typing import Tuple

import numpy
import torch
from torch import Tensor, nn

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
    logits = model(X) 
    probabilities = nn.Softmax(dim=1)(logits)
    predicted_indices = probabilities.argmax(1)
  return predicted_indices


def init():
  logging.info('Init started')

  global model
  global device

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './outputs/weights.pth')
  # Replace previous line with next line and uncomment main to test locally. 
  # model_path = './outputs/weights.pth'

  model = NeuralNetwork().to(device)
  model.load_state_dict(torch.load(model_path))
  model.eval()

  logging.info('Init complete')


def run(raw_data):
  logging.info('Request received')

  X = json.loads(raw_data)['data']
  X = numpy.array(X)
  X = torch.from_numpy(X).float().to(device)
  
  predicted_indices = predict(model, X)
  predicted_names = [labels_map[predicted_index.item()] for predicted_index in predicted_indices]

  logging.info(f'Predicted names: {predicted_names}')

  logging.info('Request processed')
  return predicted_names


# if __name__ == '__main__':
#   init()
#   with open('outputs/sample_request.json') as file:
#     raw_data = file.read()
#   print(run(raw_data))  