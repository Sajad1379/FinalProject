import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import openpyxl
import os
import time

# Add_laptop = "C:\Users\sajad\Desktop\FinalProject"

# Add_comp = "C:\Users\hp\Desktop\FinalProject"

studentVle=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\studentVleTest.xlsx',engine='openpyxl')

student=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\testData.xlsx',engine='openpyxl')

dataMain = pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\20perTran.xlsx',engine='openpyxl')

rows_studentVle, cols_studentVle = studentVle.shape
print(f"Number of rows: {rows_studentVle}")
print(f"Number of columns: {cols_studentVle}")

print('================================')
rows_student, cols_student = student.shape
print(f"Number of rows: {rows_student}")
print(f"Number of columns: {cols_student}")

#==========================================================
weight_student = []

for i in range(0,rows_student):
    weight_student.append([student.iloc[i,0],student.iloc[i,1]])


X_student= sorted(weight_student, key=lambda x: x[0])



#============================================================


weight_studentVle = []

for i in range(0,rows_studentVle):
    weight_studentVle.append([studentVle.iloc[i,2],studentVle.iloc[i,5],studentVle.iloc[i,6]])


X_studentVle = sorted(weight_studentVle, key=lambda x: x[0])



#============================================================


start = time.time()

count = 0
for i in range(0 , rows_student):
    for j in range(0 , rows_studentVle):
        if(X_student[i][0] == X_studentVle[j][0]):
            print("yesss")
            exit(0)
            dataMain.at[count , 'id_student' ] = X_studentVle[j][0]
            dataMain.at[count , 'activity_type' ] = X_studentVle[j][2]
            dataMain.at[count , 'sum_click' ] = X_studentVle[j][1]
            count = count + 1

end = time.time()

print(end-start)
dataMain.to_excel(r'C:\Users\sajad\Desktop\FinalProject\20perTran.xlsx', index=False)
