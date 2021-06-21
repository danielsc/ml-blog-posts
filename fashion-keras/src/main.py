import os
import random
from typing import Tuple, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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


def get_data(batch_size) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  (training_data, training_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

  training_data = training_data / 255.0
  test_data = test_data / 255.0

  train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels)).batch(batch_size).shuffle(500)
  test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size).shuffle(500)

  return (train_dataset, test_dataset)


def visualize_data(data) -> None:
  figure = plt.figure(figsize=(8, 8))
  cols, rows = 3, 3
  for i in range(1, cols * rows + 1):
      sample_idx = random.randint(0, len(data[0]))
      image = data[0][sample_idx]
      label = data[1][sample_idx]
      figure.add_subplot(rows, cols, i)
      plt.title(labels_map[label])
      plt.axis('off')
      plt.imshow(image.squeeze(), cmap='gray')
  plt.show()


def get_model() -> tf.keras.Sequential:
  model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
  ])
  return model


def training_phase():
  learning_rate = 0.1
  batch_size = 64
  epochs = 2

  (train_dataset, test_dataset) = get_data(batch_size)
  # visualize_data(training_data)

  model = get_model()
  # model.summary()

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.optimizers.SGD(learning_rate)
  metrics = ['accuracy']
  model.compile(optimizer, loss_fn, metrics)

  print('\nFitting:')
  model.fit(train_dataset, epochs=epochs)
    
  print('\nEvaluating:')
  (test_loss, test_accuracy) = model.evaluate(test_dataset)
  print(f'\nTest accuracy: {test_accuracy * 100:>0.1f}%, test loss: {test_loss:>8f}')

  model.save_weights('outputs/weights')


def predict(model: tf.keras.Model, X: np.ndarray) -> tf.Tensor:
  logits = model(X)
  probabilities = tf.keras.layers.Softmax(axis=1)(logits)
  predicted_index = tf.math.argmax(input=probabilities, axis=1)
  return predicted_index


def inference_phase():
  batch_size = 64

  model = get_model()
  model.load_weights('outputs/weights').expect_partial()

  (_, test_dataset) = get_data(batch_size)
  (X_batch, actual_index_batch) = next(test_dataset.as_numpy_iterator())
  X = X_batch[0:3, :, :]
  actual_indices = actual_index_batch[0:3]

  predicted_indices = predict(model, X)

  print('\nPredicting:')
  for (actual_index, predicted_index) in zip(actual_indices, predicted_indices):
    actual_name = labels_map[actual_index]
    predicted_name = labels_map[predicted_index.numpy()]
    print(f'Actual: {actual_name}, Predicted: {predicted_name}')


def main() -> None:
  training_phase()
  inference_phase()


if __name__ == '__main__':
  main()
