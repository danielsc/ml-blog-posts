import logging
import os

import numpy as np
from PIL import Image
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
  logging.info("Num GPUs:", len(physical_devices))

  model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'tf_model')
  # Replace previous line with next line and uncomment main to test locally. 
  # model_path = './tf_model'
  model = tf.keras.models.load_model(model_path)

  logging.info('Init complete')
  pass


def run(mini_batch):
  logging.info(f'Run started: {__file__}, run({mini_batch}')
  predicted_names = []

  for image_path in mini_batch:
    image = Image.open(image_path)
    array = tf.keras.preprocessing.image.img_to_array(image).reshape((-1, 28, 28))
    predicted_index = predict(model, array).numpy().sum()
    predicted_names.append(labels_map[predicted_index])

  logging.info('Run completed')
  return predicted_names


# if __name__ == '__main__':
#   init()
#   image_paths = [f'sample_request/{filename}' for filename in os.listdir('sample_request')]
#   image_paths.sort()
#   print(run(image_paths))  