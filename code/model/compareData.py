import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

original_data = pd.read_csv('C:\\Users\\malik\\Desktop\\PateGan-DeepLearning\\data\\Output\\30_05_2024_example2\\ori_data.csv')
synthetic_data = pd.read_csv('C:\\Users\\malik\\Desktop\\PateGan-DeepLearning\\data\\Output\\30_05_2024_example2\\synthetic_data.csv')

feature_names = [f"feature_{i}" for i in range(14)]

df_original = pd.DataFrame(original_data, columns=feature_names)
df_synthetic = pd.DataFrame(synthetic_data, columns=feature_names)

for feature in feature_names:
    plt.figure(figsize=(10, 5))
    sns.histplot(df_original[feature], color='blue', kde=True, label='Orijinal Veri', stat="density", linewidth=0)
    sns.histplot(df_synthetic[feature], color='orange', kde=True, label='Sentetik Veri', stat="density", linewidth=0)
    plt.title(f'{feature} Karşılaştırması')
    plt.xlabel(f'{feature}')
    plt.ylabel('Yoğunluk')
    plt.legend()
    plt.show()
