import json
from train import get_data 

def create_sample_request() -> None:
  batch_size = 64
  (_, test_dataloader) = get_data(batch_size)
  
  (X_batch, _) = next(iter(test_dataloader))
  X = X_batch[0:3, :, :, :].cpu().numpy().tolist()
  with open('sample_request/sample_request.json', 'w') as file:
    json.dump({ 'data': X }, file)


def main() -> None:
  create_sample_request()


if __name__ == '__main__':
  main()
