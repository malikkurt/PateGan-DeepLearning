import pandas as pd

from evaluation_in_prod import eval_metrics
# import numpy as np
# from datetime import datetime
# import os
# import argparse
# from main_pategan_experiment import pategan_main  # pategan_main fonksiyonunun bulunduğu scriptin adıyla değiştirin

# if __name__ == '__main__':
#     # Varsayılan değerler
#     input_dir = "C:\\Users\\malik\\Desktop\\PateGan-DeepLearning\\data\\Input"  # Buraya input klasörünüzün yolunu yazın
#     base_output_dir = "data/Output"
#     today = datetime.today().strftime("%d_%m_%Y")

#     # Parser tanımlaması ve argümanların eklenmesi
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_no', default=10000, type=int)
#     parser.add_argument('--data_dim', default=10, type=int)
#     parser.add_argument('--dataset', default='csvfile', type=str)
#     parser.add_argument('--noise_rate', default=1.0, type=float)
#     parser.add_argument('--iterations', default=50, type=int)
#     parser.add_argument('--n_s', default=1, type=int)
#     parser.add_argument('--batch_size', default=64, type=int)
#     parser.add_argument('--k', default=10, type=float)
#     parser.add_argument('--epsilon', default=1.0, type=float)
#     parser.add_argument('--delta', default=0.00001, type=float)
#     parser.add_argument('--lamda', default=1.0, type=float)

#     args = parser.parse_args()

#     for root, dirs, files in os.walk(input_dir):
#         for file in files:
#             if file.endswith('.csv'):
#                 csv_file = os.path.join(root, file)
#                 print(f'Processing {csv_file}...')

#                 # pategan_main fonksiyonunu çağırma
#                 results, ori_data, synth_data = pategan_main(args,csv_file)

#                 # Çıktı dizinini oluşturma
#                 output_dir = os.path.join(base_output_dir, f"{today}_{os.path.splitext(file)[0]}")
#                 os.makedirs(output_dir, exist_ok=True)

#                 # Orijinal veri DataFrame'ini oluşturma ve kaydetme
#                 ori_data_df = pd.DataFrame(ori_data, columns=[f"feature_{i}" for i in range(ori_data.shape[1])])
#                 ori_data_path = os.path.join(output_dir, "ori_data.csv")
#                 ori_data_df.to_csv(ori_data_path, index=False)

#                 # Sentezlenmiş veri DataFrame'ini oluşturma ve kaydetme
#                 synth_data_df = pd.DataFrame(synth_data, columns=[f"feature_{i}" for i in range(synth_data.shape[1])])
#                 synth_data_path = os.path.join(output_dir, "synthetic_data.csv")
#                 synth_data_df.to_csv(synth_data_path, index=False)

#                 print(f'Finished processing {csv_file}.')












# Verilen orijinal ve sentetik verileri okuyalım
original_data = pd.read_csv("C:\\Users\\malik\\Desktop\\PateGan-DeepLearning\\data\\Output\\30_05_2024_example2\\ori_data.csv")
synthetic_data = pd.read_csv("C:\\Users\\malik\\Desktop\\PateGan-DeepLearning\\data\\Output\\30_05_2024_example2\\synthetic_data.csv")

# Metotları uygulayalım
metrics = eval_metrics(original_data, synthetic_data)

# Kolmogorov-Smirnov Testi
ks_results = metrics.kolmogorov()
print("Kolmogorov-Smirnov Test Results:", ks_results)

# # # Jensen-Shannon Divergence
# # js_results = metrics.jensen_shannon()
# # print("Jensen-Shannon Divergence Results:", js_results)

# # Kullback-Leibler Divergence
# kl_results = metrics.kl_divergence()
# print("Kullback-Leibler Divergence Results:", kl_results)

# # # Pairwise Correlation Difference
# # pcd_result = metrics.pairwise_correlation_difference()
# # print("Pairwise Correlation Difference Result:", pcd_result)
