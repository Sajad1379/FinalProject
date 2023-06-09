import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import openpyxl
import os

studentVle=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\studentVleTest.xlsx',engine='openpyxl')

vle=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\vle.xlsx',engine='openpyxl')


rows_studentVle, cols_studentVle = studentVle.shape
print(f"Number of rows: {rows_studentVle}")
print(f"Number of columns: {cols_studentVle}")

print('================================')

rows_vle, cols_vle = vle.shape
print(f"Number of rows_s: {rows_vle}")
print(f"Number of columns_s: {cols_vle}")

print('================================')

weight_vle = []

for i in range(0,rows_vle):
    weight_vle.append([vle.iloc[i,0],vle.iloc[i,3]])

X_vle=np.array(weight_vle)


weight_studentVle = []
for i in range(0,rows_studentVle):
    weight_studentVle.append([studentVle.iloc[i,3]])
    
X_studentVle=np.array(weight_studentVle , dtype=np.str_)
print(X_studentVle)


for i in range(0,rows_studentVle):
    for j in range(0,rows_vle):
        if(X_studentVle[i][0] == X_vle[j][0]):
            studentVle.at[i , "activity_type"] = X_vle[j][1]
            break

studentVle.to_excel(r'C:\Users\sajad\Desktop\FinalProject\studentVleTest.xlsx', index=False)



