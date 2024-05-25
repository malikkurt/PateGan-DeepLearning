import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.special import rel_entr
from scipy.stats import ks_2samp
from evaluation_in_prod import eval_metrics

# Verilen orijinal ve sentetik verileri okuyalım
original_data = pd.read_csv("C:\\Users\\malik\\Desktop\\PateGan-DeepLearning\\data\\Output\\ori_data.csv")
synthetic_data = pd.read_csv("C:\\Users\\malik\\Desktop\\PateGan-DeepLearning\\data\\Output\\synthetic_data.csv")

# Metotları uygulayalım
metrics = eval_metrics(original_data, synthetic_data)

# Kolmogorov-Smirnov Testi
ks_results = metrics.kolmogorov()
print("Kolmogorov-Smirnov Test Results:", ks_results)

# # Jensen-Shannon Divergence
# js_results = metrics.jensen_shannon()
# print("Jensen-Shannon Divergence Results:", js_results)

# Kullback-Leibler Divergence
kl_results = metrics.kl_divergence()
print("Kullback-Leibler Divergence Results:", kl_results)

# # Pairwise Correlation Difference
# pcd_result = metrics.pairwise_correlation_difference()
# print("Pairwise Correlation Difference Result:", pcd_result)
