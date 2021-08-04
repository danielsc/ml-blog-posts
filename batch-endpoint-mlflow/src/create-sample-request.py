import os

import numpy as np
from torchvision import datasets


def main() -> None:
  test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
  )

  try:
    os.mkdir('sample_request')
  except:
    pass
  delimiter = ','
  fmt = '%.6f'
  header = delimiter.join([f'col_{i}' for i in range(28*28)])
  for i, (image, _) in enumerate(test_data):
    if i == 1000:
      break
    filename = f'sample_request/{i+1:0>4}.csv'
    image_array = np.asarray(image).reshape((1, -1)) / 255
    np.savetxt(filename, image_array, comments='', delimiter=delimiter, fmt=fmt, header=header)


if __name__ == '__main__':
  main()
