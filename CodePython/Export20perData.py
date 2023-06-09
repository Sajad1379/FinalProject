import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import openpyxl
import os
import time

# Add_laptop = "C:\Users\sajad\Desktop\FinalProject"

# Add_comp = "C:\Users\hp\Desktop\FinalProject"

data=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\data.xlsx',engine='openpyxl')

student=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\studentInfo.xlsx',engine='openpyxl')

testData = pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\testData.xlsx',engine='openpyxl')


rows_data, cols_data = data.shape
print(f"Number of rows: {rows_data}")
print(f"Number of columns: {cols_data}")

print('================================')
rows_student, cols_student = student.shape
print(f"Number of rows: {rows_student}")
print(f"Number of columns: {cols_student}")



#==========================================================
weight_student = []

for i in range(0,rows_student):
    weight_student.append([student.iloc[i,2],student.iloc[i,11]])


X_student= sorted(weight_student, key=lambda x: x[0])



#============================================================


weight_data = []

for i in range(0,rows_data):
    weight_data.append([data.iloc[i,0]])


X_data = np.array(weight_data)

X_student= sorted(weight_data, key=lambda x: x[0])

#============================================================
count =0
for i in range(0,len(X_student)):
    for j in range(0,rows_data):
        if(i >= len(X_student)):
            break
        if((X_data[j][0]) == (X_student[i][0])):
            del X_student[i]
            break
    
print(len(X_student))
for i in range(0 , len(X_student)):
    testData.at[count , 'student_id'] = X_student[i][0]
    testData.at[count , 'final_result'] = X_student[i][1]

testData.to_excel(r'C:\Users\sajad\Desktop\FinalProject\testData.xlsx', index=False)
