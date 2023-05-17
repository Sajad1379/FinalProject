import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import openpyxl
import os

data=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\data.xlsx',engine='openpyxl')

data1=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\studentInfo.xlsx',engine='openpyxl')


rows_d, cols_d = data.shape
print(f"Number of rows: {rows_d}")
print(f"Number of columns: {cols_d}")

print('================================')

rows_s, cols_s = data1.shape
print(f"Number of rows_s: {rows_s}")
print(f"Number of columns_s: {cols_s}")

print('================================')

weight = []

for i in range(0,rows_s):
    weight.append([data1.iloc[i,2],data1.iloc[i,11]])

X=np.array(weight)


weight1 = []
for i in range(0,rows_d):
    weight1.append([data.iloc[i,0]])
    
X1=np.array(weight1 , dtype=np.str_)
print(X1)


for i in range(0,rows_d):
    flag = true;
    for j in range(0,rows_s):
        if(X1[i][0] == X[j][0]):
            flag = false
            break
    if(flag == false):
        
        

data.to_excel(r'C:\Users\sajad\Desktop\FinalProject\data.xlsx', index=False)
















