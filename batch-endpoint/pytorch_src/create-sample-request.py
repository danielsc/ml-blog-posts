from torchvision import datasets
import os

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
  for i, (image, _) in enumerate(test_data):
    if i == 1000:
      break
    image.save(f'sample_request/{i+1:0>4}.png')


if __name__ == '__main__':
  main()
