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
    weight.append([data.iloc[i,5],data.iloc[i,7],data.iloc[i,10],data.iloc[i,15]])
    
X=np.array(weight)

kmeans = KMeans(n_clusters=4, n_init=206)

kmeans.fit(X)

data_c=kmeans.labels_

for i in range(0,rows):
    data.at[i , "k-means"]=data_c[i]



data.to_excel(r'C:\Users\sajad\Desktop\FinalProject\data.xlsx', index=False)
