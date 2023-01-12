import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


total = np.zeros(8)  #初始化feature數量
for i in range(1,10):
    df = pd.read_csv('../data/Data_0_24/place_'+ str(i) +'.csv')
    df.drop('date', inplace=True, axis=1)

    plt.figure(figsize=(11,7))
    sns.heatmap(df.corr(), annot = True)
    plt.show()
    plt.savefig('../data_process/correlation_heatmap.jpg')

    total = total + np.array(df.corr()['OT'])

print(total/9)


