import os
import time
from typing import Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def get_data(batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

  train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

  train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label))
  test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label))

  train_dataset = train_dataset.batch(batch_size).shuffle(500)
  test_dataset = test_dataset.batch(batch_size).shuffle(500)

  return (train_dataset, test_dataset)


@tf.function
def fit_one_batch(X: tf.Tensor, y: tf.Tensor, model: tf.keras.Model, loss_fn: tf.keras.losses.Loss, 
optimizer: tf.keras.optimizers.Optimizer) -> Tuple[tf.Tensor, tf.Tensor]:
  with tf.GradientTape() as tape:
    y_prime = model(X, training=True)
    loss = loss_fn(y, y_prime)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return (y_prime, loss)


def fit(dataset: tf.data.Dataset, model: tf.keras.Model, loss_fn: tf.keras.losses.Loss, 
optimizer: tf.optimizers.Optimizer) -> None:
  batch_count = len(dataset)
  loss_sum = 0
  correct_item_count = 0
  current_item_count = 0
  print_every = 100

  for batch_index, (X, y) in enumerate(dataset):
    (y_prime, loss) = fit_one_batch(X, y, model, loss_fn, optimizer)

    y = tf.cast(y, tf.int64)
    correct_item_count += (tf.math.argmax(y_prime, axis=1) == y).numpy().sum()

    batch_loss = loss.numpy()
    loss_sum += batch_loss
    current_item_count += len(X)

    if ((batch_index + 1) % print_every == 0) or ((batch_index + 1) == batch_count):
      batch_accuracy = correct_item_count / current_item_count * 100
      print(f'[Batch {batch_index + 1:>3d} - {current_item_count:>5d} items] accuracy: {batch_accuracy:>0.1f}%, loss: {batch_loss:>7f}')


@tf.function
def evaluate_one_batch(X: tf.Tensor, y: tf.Tensor, model: tf.keras.Model, 
loss_fn: tf.keras.losses.Loss) -> Tuple[tf.Tensor, tf.Tensor]:
  y_prime = model(X, training=False)
  loss = loss_fn(y, y_prime)

  return (y_prime, loss)


def evaluate(dataset: tf.data.Dataset, model: tf.keras.Model, 
loss_fn: tf.keras.losses.Loss) -> Tuple[float, float]:
  batch_count = len(dataset)
  loss_sum = 0
  correct_item_count = 0
  current_item_count = 0

  for (X, y) in dataset:
    (y_prime, loss) = evaluate_one_batch(X, y, model, loss_fn)

    correct_item_count += (tf.math.argmax(y_prime, axis=1).numpy() == y.numpy()).sum()
    loss_sum += loss.numpy()
    current_item_count += len(X)

  average_loss = loss_sum / batch_count
  accuracy = correct_item_count / current_item_count
  return (average_loss, accuracy)


def training_phase():
  learning_rate = 0.1
  batch_size = 64
  epochs = 2

  (train_dataset, test_dataset) = get_data(batch_size)

  model = NeuralNetwork()

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.optimizers.SGD(learning_rate)

  print('\nFitting:')
  t_begin = time.time()
  for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}\n-------------------------------')
    fit(train_dataset, model, loss_fn, optimizer)
  t_elapsed = time.time() - t_begin
  print(f'\nTime per epoch: {t_elapsed / epochs :>.3f} sec' )

  print('\nEvaluating:')
  (test_loss, test_accuracy) = evaluate(test_dataset, model, loss_fn)
  print(f'Test accuracy: {test_accuracy * 100:>0.1f}%, test loss: {test_loss:>8f}')

  model.save_weights('tf_model/weights')


def main() -> None:
  training_phase()


if __name__ == '__main__':
  main()
