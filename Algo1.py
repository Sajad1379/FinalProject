import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import openpyxl
import os

data=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\data.xlsx',engine='openpyxl')

rows, cols = data.shape
print(f"Number of rows: {rows}")
print(f"Number of columns: {cols}")

print('================================')
weight = []
for i in range(0,rows):
    for j in range(1,20):
        weight.append([data.iloc[i,j]])
    
X=np.array(weight)
print(X)
kmeans = KMeans(n_clusters=5, n_init=206)

kmeans.fit(X)

data_c=kmeans.labels_

for i in range(0,rows):
    data.at[i , "k-means"]=data_c[i]



data.to_excel(r'C:\Users\sajad\Desktop\FinalProject\data.xlsx', index=False)
