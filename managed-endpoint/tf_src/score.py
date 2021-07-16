import json
import logging
import os

import numpy as np
import tensorflow as tf

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


@tf.function
def predict(model: tf.keras.Model, X: np.ndarray) -> tf.Tensor:
  y_prime = model(X, training=False)
  probabilities = tf.nn.softmax(y_prime, axis=1)
  predicted_indices = tf.math.argmax(input=probabilities, axis=1)
  return predicted_indices


def init():
  logging.info('Init started')

  global model
  global device

  # device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # logging.info(f'Device: {device}')
  model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'weights.pth')

  model = NeuralNetwork()
  model.load_weights(model_path)

  logging.info('Init complete')
  pass


def run(raw_data):
  logging.info('Run started')

  X = json.loads(raw_data)['data']
  X = np.array(X)
  
  predicted_indices = predict(model, X)
  predicted_names = [labels_map[predicted_index.item()] for predicted_index in predicted_indices]

  logging.info(f'Predicted names: {predicted_names}')

  logging.info('Run completed')
  return predicted_names
