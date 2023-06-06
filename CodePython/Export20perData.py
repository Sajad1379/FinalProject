import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import openpyxl
import os
import time

# Add_laptop = "C:\Users\sajad\Desktop\FinalProject"

# Add_comp = "C:\Users\hp\Desktop\FinalProject"

data=pd.read_excel(r'C:\Users\hp\Desktop\FinalProject\data.xlsx',engine='openpyxl')

student=pd.read_excel(r'C:\Users\hp\Desktop\FinalProject\studentInfo.xlsx',engine='openpyxl')

testData = pd.read_excel(r'C:\Users\hp\Desktop\FinalProject\testData.xlsx',engine='openpyxl')


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


X_student= np.array(weight_student)



#============================================================


weight_data = []

for i in range(0,rows_data):
    weight_data.append([data.iloc[i,0]])


X_data = np.array(weight_data)



#============================================================
count =0
for i in range(0,rows_student):
    flag = True
    for j in range(0,rows_data):
        if(int(X_data[j][0]) == int(X_student[i][0])):

            flag = False
            break
    if(flag):
        print(X_student[i][0])
        testData.at[count , 'student_id'] = X_student[i][0]
        testData.at[count , 'final_result'] = X_student[i][1]
        count = count + 1

testData.to_excel(r'C:\Users\hp\Desktop\FinalProject\testData.xlsx', index=False)
