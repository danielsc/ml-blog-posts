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

  physical_devices = tf.config.list_physical_devices('GPU')
  logging.info(f'Num GPUs: {len(physical_devices)}')

  model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'tf_model/weights')
  # Replace previous line with next line and uncomment main to test locally. 
  # model_path = './tf_model/weights'

  model = NeuralNetwork()
  model.load_weights(model_path)

  logging.info('Init complete')
  pass


def run(raw_data):
  logging.info('Run started')

  X = json.loads(raw_data)['data']
  X = np.array(X).reshape((-1, 28, 28))
  
  predicted_index = predict(model, X).numpy()[0]
  predicted_name = labels_map[predicted_index]

  logging.info(f'Predicted name: {predicted_name}')

  logging.info('Run completed')
  return predicted_name


# if __name__ == '__main__':
#   init()
#   with open('sample_request/sample_request.json') as file:
#     raw_data = file.read()
#   print(run(raw_data))  