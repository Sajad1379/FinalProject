import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import openpyxl
import os
import time

# Add_laptop = "C:\Users\sajad\Desktop\FinalProject"

# Add_comp = "C:\Users\hp\Desktop\FinalProject"


activity_type = [
    'dataplus','dualpane','externalquiz',
    'folder' , 'forumng' , 'glossary',
    'homepage' , 'htmlactivity' , 'oucollaborate',
    'ouwiki' , 'page' , 'questionnaire' , 'quiz'
    'repeatactivity' , 'resource' , 'sharedsubpage',
    'subpage' , 'url'    
 ]

studentVle=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\studentVleTest.xlsx',engine='openpyxl')

student=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\test.xlsx',engine='openpyxl')

dataMain = pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\DataMain.xlsx',engine='openpyxl')

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
    weight_student.append([student.iloc[i,2],student.iloc[i,11]])


X_student= sorted(weight_student, key=lambda x: x[0])



#============================================================


weight_studentVle = []

for i in range(0,rows_studentVle):
    weight_studentVle.append([studentVle.iloc[i,2],studentVle.iloc[i,5],studentVle.iloc[i,6]])


X_studentVle = sorted(weight_studentVle, key=lambda x: x[0])



#============================================================


start = time.time()




for i in range(0 , 1):
    dataMain.at[i , 'id_student'] = X_student[i][0]
    for j in range(0,len(activity_type)):
        sum = 0
        count = 0
        for k in range(0,rows_studentVle):
            
            if(int(X_student[i][0]) == int(X_studentVle[k][0]) and str(activity_type[j])==str(X_studentVle[k][2])):
                
                count = count + 1
                sum = sum + int(X_studentVle[k][1])
            
        dataMain.at[i , activity_type[j]] = sum
    
    dataMain.at[i , 'final_result'] = X_student[i][1]
end = time.time()

print(end-start)
dataMain.to_excel(r'C:\Users\sajad\Desktop\FinalProject\DataMain.xlsx', index=False)
