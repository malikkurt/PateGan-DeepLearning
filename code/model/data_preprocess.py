import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def data_preprocess(path):

      data = pd.read_csv(path).to_numpy()
      data = MinMaxScaler().fit_transform(data)
      train_ratio = 0.5
      train = np.random.rand(data.shape[0])<train_ratio 
      train_data, test_data = data[train], data[~train]
      data_dim = data.shape[1]

      return data_dim,train_data,test_data


