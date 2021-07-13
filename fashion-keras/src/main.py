import os
import random
from typing import Tuple

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


def get_data(batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

  train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

  train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label))
  test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label))

  train_dataset = train_dataset.batch(batch_size).shuffle(500)
  test_dataset = test_dataset.batch(batch_size).shuffle(500)

  return (train_dataset, test_dataset)


def visualize_data(dataset: tf.data.Dataset) -> None:
  first_batch = dataset.as_numpy_iterator().next()
  figure = plt.figure(figsize=(8, 8))
  cols = 3
  rows = 3
  for i in range(1, cols * rows + 1):
    sample_idx = random.randint(0, len(first_batch[0]))
    image = first_batch[0][sample_idx]
    label = first_batch[1][sample_idx]
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
  # visualize_data(train_dataset)

  model = get_model()
  # model.summary()

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.SGD(learning_rate)
  metrics = ['accuracy']
  model.compile(optimizer, loss_fn, metrics)

  print('\nFitting:')
  model.fit(train_dataset, epochs=epochs)
    
  print('\nEvaluating:')
  (test_loss, test_accuracy) = model.evaluate(test_dataset)
  print(f'\nTest accuracy: {test_accuracy * 100:>0.1f}%, test loss: {test_loss:>8f}')

  model.save_weights('outputs/weights')


def inference_phase():
  batch_size = 64

  model = get_model()
  model.load_weights('outputs/weights').expect_partial()

  (_, test_dataset) = get_data(batch_size)
  (X_batch, actual_index_batch) = test_dataset.as_numpy_iterator().next()
  X = X_batch[0:3, :, :]
  actual_indices = actual_index_batch[0:3]

  predicted_indices = model.predict(X)

  print('\nPredicting:')
  for (actual_index, predicted_index) in zip(actual_indices, predicted_indices):
    actual_name = labels_map[actual_index]
    predicted_name = labels_map[tf.math.argmax(predicted_index).numpy()]
    print(f'Actual: {actual_name}, Predicted: {predicted_name}')


def main() -> None:
  training_phase()
  inference_phase()


if __name__ == '__main__':
  main()
