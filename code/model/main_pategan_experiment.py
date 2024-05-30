from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')

from data_preprocess import data_preprocess
from utils import supervised_model_training
from pate_gan import pategan
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from evaluation_in_prod import eval_metrics


#%% 
def pategan_main (args,filePath):
  
  models = ['logisticregression','randomforest', 'gaussiannb','bernoullinb',
            'svmlin', 'Extra Trees','LDA', 'AdaBoost','Bagging','gbm', 'xgb']
  
  data_dim,train_data,test_data = data_preprocess(filePath)
  
  results = np.zeros([len(models), 4])
  
  parameters = {'n_s': args.n_s, 'batch_size': args.batch_size, 'k': args.k, 
                'epsilon': args.epsilon, 'delta': args.delta, 
                'lamda': args.lamda}
  
  # Generate synthetic training data
  best_perf = 0.0
    
  for it in range(args.iterations):
    print('Iteration',it)
    synth_train_data_temp = pategan(train_data, parameters)
    temp_perf, _ = supervised_model_training(
        synth_train_data_temp[:, :(data_dim-1)], 
        np.round(synth_train_data_temp[:, (data_dim-1)]),
        train_data[:, :(data_dim-1)], 
        np.round(train_data[:, (data_dim-1)]),
        'logisticregression')
    
    # Select best synthetic data
    if temp_perf > best_perf:
      best_perf = temp_perf.copy()
      synth_train_data = synth_train_data_temp.copy()
      
    print('Iteration: ' + str(it+1))
    print('Best-Perf:' + str(best_perf))
  
  # Train supervised models
  for model_index in range(len(models)):
    model_name = models[model_index]
    
    # Using original data
    results[model_index, 0], results[model_index, 2] = (
        supervised_model_training(train_data[:, :(data_dim-1)], 
                                  np.round(train_data[:, (data_dim-1)]),
                                  test_data[:, :(data_dim-1)], 
                                  np.round(test_data[:, (data_dim-1)]),
                                  model_name))
        
    # Using synthetic data
    results[model_index, 1], results[model_index, 3] = (
        supervised_model_training(synth_train_data[:, :(data_dim-1)], 
                                  np.round(synth_train_data[:, (data_dim-1)]),
                                  test_data[:, :(data_dim-1)], 
                                  np.round(test_data[:, (data_dim-1)]),
                                  model_name))

    
    
  # Print the results for each iteration
  results = pd.DataFrame(np.round(results, 4), 
                         columns=['AUC-Original', 'AUC-Synthetic', 
                                  'APR-Original', 'APR-Synthetic'])
  print(results)
  print('Averages:')
  print(results.mean(axis=0))
  
  return results, train_data, synth_train_data

  
# %%

#%%  
if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_no',
      help='number of generated data',
      default=10000,
      type=int)
  parser.add_argument(
      '--data_dim',
      help='number of dimensions of generated dimension (if random)',
      default=10,
      type=int)
  parser.add_argument(
      '--dataset',
      help='dataset to use',
      default='csvfile',
      type=str)
  parser.add_argument(
      '--noise_rate',
      help='noise ratio on data',
      default=1.0,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of iterations for handling initialization randomness',
      default=50,
      type=int)
  parser.add_argument(
      '--n_s',
      help='the number of student training iterations',
      default=1,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of batch size for training student and generator',
      default=64,
      type=int)
  parser.add_argument(
      '--k',
      help='the number of teachers',
      default=10,
      type=float)
  parser.add_argument(
      '--epsilon',
      help='Differential privacy parameters (epsilon)',
      default=1.0,
      type=float)
  parser.add_argument(
      '--delta',
      help='Differential privacy parameters (delta)',
      default=0.00001,
      type=float)
  parser.add_argument(
      '--lamda',
      help='PATE noise size',
      default=0.1,
      type=float)
  
  args = parser.parse_args()    
  
  results, ori_data, synth_data = pategan_main(args,"C:\\Users\\malik\\Desktop\\PateGan-DeepLearning\\data\\Input\\dataset-1\\adult_labels.csv")

  today = datetime.today().strftime("%d_%m_%Y")

  base_output_dir = "data/Output"
  csv_filename = "example2.csv" 
  output_dir = os.path.join(base_output_dir, f"{today}_{os.path.splitext(csv_filename)[0]}")
  os.makedirs(output_dir, exist_ok=True)

  # Orijinal veri DataFrame'ini oluştur
  ori_data_df = pd.DataFrame(ori_data, columns=[f"feature_{i}" for i in range(len(ori_data[0]))])
  ori_data_path = os.path.join(output_dir, "ori_data.csv")
  ori_data_df.to_csv(ori_data_path, index=False)

  # Sentezlenmiş veri DataFrame'ini oluştur
  synth_data_df = pd.DataFrame(synth_data, columns=[f"feature_{i}" for i in range(len(synth_data[0]))])
  synth_data_path = os.path.join(output_dir, "synthetic_data.csv")
  synth_data_df.to_csv(synth_data_path, index=False)


















#   # Çıktı dizinini oluştur
#   base_output_dir = "data/Output"
#   csv_filename = "example2.csv"  # CSV dosyasının adı (gerçek dosya adınızı buraya koyun)
#   output_dir = os.path.join(base_output_dir, f"{today}_{os.path.splitext(csv_filename)[0]}")
#   os.makedirs(output_dir, exist_ok=True)

#   # Orijinal veri DataFrame'ini oluştur
#   ori_data_df = pd.DataFrame(ori_data, columns=[f"feature_{i}" for i in range(len(ori_data[0]))])
#   ori_data_path = os.path.join(output_dir, "ori_data.csv")
#   ori_data_df.to_csv(ori_data_path, index=False)

#   # Sentezlenmiş veri DataFrame'ini oluştur
#   synth_data_df = pd.DataFrame(synth_data, columns=[f"feature_{i}" for i in range(len(synth_data[0]))])
#   synth_data_path = os.path.join(output_dir, "synthetic_data.csv")
#   synth_data_df.to_csv(synth_data_path, index=False)

#   metrics = eval_metrics(ori_data_df, synth_data_df)
#   ks_results = metrics.kolmogorov()

#   # Test sonuçlarını CSV dosyasına yaz
#   ks_results_path = os.path.join(output_dir, "ks_results.csv")
#   ks_results.to_csv(ks_results_path, index=False)
# %%
