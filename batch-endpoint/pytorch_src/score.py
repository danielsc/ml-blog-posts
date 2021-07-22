import argparse
import logging
import os

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms


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
  global logger 
  global model
  global device

  arg_parser = argparse.ArgumentParser(description="Argument parser.")
  arg_parser.add_argument("--logging_level", type=str, help="logging level")
  args, unknown_args = arg_parser.parse_known_args()  
  logger = logging.getLogger(__name__)
  logger.setLevel(args.logging_level.upper())

  logger.info('*** Init started')

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  logger.info(f'Device: {device}')

  model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pth')
  # Replace previous line with next line and uncomment main to test locally. 
  # model_path = './pytorch_model/model.pth'

  model = torch.load(model_path, map_location=device)
  model.eval()

  logger.info('*** Init complete')
  pass


def run(mini_batch):
  logger.info(f'*** Run started')
  predicted_names = []
  transform = transforms.ToTensor()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  for image_path in mini_batch:
    image = Image.open(image_path)
    tensor = transform(image).to(device)
    predicted_index = predict(model, tensor).item()
    predicted_names.append(labels_map[predicted_index])

  logger.info('*** Run completed')
  return predicted_names


# if __name__ == '__main__':
#   init()
#   image_paths = [f'sample_request/{filename}' for filename in os.listdir('sample_request')]
#   image_paths.sort()
#   print(run(image_paths))  
