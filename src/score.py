
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
  global model
  model_path = Model.get_model_path('model')
  model = joblib.load(model_path)

def run(raw_data):
  try: ## Try la predicci�n.
    data = json.loads(raw_data)['data'][0]
    data = pd.DataFrame(data)


    result = model.predict(data).tolist()
    result_sigmoid = np.sigmoid(result)
    umbral = 0.5260736508016486
    result_finals = [1 if x > umbral else 0 for x in result_sigmoid]

    return json.dumps(result_finals)
  except Exception as e:
    return json.dumps(str(e))
