from torchvision import datasets
import os

# def create_sample_request() -> None:
#   batch_size = 64
#   (_, test_dataloader) = get_data(batch_size)
  
#   (X_batch, _) = next(iter(test_dataloader))
#   X = X_batch[0:3, :, :, :].cpu().numpy().tolist()
#   with open('sample_request/sample_request.json', 'w') as file:
#     json.dump({ 'data': X }, file)


# def main() -> None:
#   create_sample_request()

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
