import mlflow
import pandas as pd
import torch

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


class ModelWrapper(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    self.model = mlflow.pytorch.load_model(context.artifacts['pytorch-model'])

  def predict(self, context, model_input: pd.DataFrame):
    with torch.no_grad():
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      tensor_input = torch.Tensor(model_input.values).to(device)
      y_prime = self.model(tensor_input)
      probabilities = torch.nn.functional.softmax(y_prime, dim=1)
      predicted_indices = probabilities.argmax(1)
      predicted_names = [labels_map[predicted_index.item()] for predicted_index
        in predicted_indices]
    return predicted_names
