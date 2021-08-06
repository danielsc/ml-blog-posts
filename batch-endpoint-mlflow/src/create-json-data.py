import os

import numpy as np
from torchvision import datasets
import pandas as pd


def main() -> None:
  test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
  )

  try:
    os.mkdir('json_data')
  except:
    pass


  for i in range(10):
    images = np.asarray(test_data.data[(i*100):(i+1)*100]).reshape((100, 1, -1)).squeeze() / 255
    df = pd.DataFrame(data=images)
    filename = f'json_data/fashion_mnist_{i*100:04}_to_{(i+1)*100:04}.json'
    print('writing file:', filename)
    df.to_json(filename, orient='split') 

if __name__ == '__main__':
  main()
