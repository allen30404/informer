import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('../data/Data_6_18/site_north_complete.csv')
df.drop('date', inplace=True, axis=1)
plt.figure(figsize=(11,7))
sns.heatmap(df.corr(method='spearman'), annot = True)
plt.show()
plt.savefig('../data_process/correlation_heatmap_north.jpg')