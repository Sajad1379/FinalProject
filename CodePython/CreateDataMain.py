import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import openpyxl
import os
import time



activity_type = [
    'dataplus','dualpane','externalquiz',
    'folder' , 'forumng' , 'glossary',
    'homepage' , 'htmlactivity' , 'oucollaborate',
    'ouwiki' , 'page' , 'questionnaire' , 'quiz'
    'repeatactivity' , 'resource' , 'sharedsubpage',
    'subpage' , 'url'    
 ]

studentVle=pd.read_excel(r'C:\Users\hp\Desktop\FinalProject\studentVleTest.xlsx',engine='openpyxl')

student=pd.read_excel(r'C:\Users\hp\Desktop\FinalProject\test.xlsx',engine='openpyxl')

dataMain = pd.read_excel(r'C:\Users\hp\Desktop\FinalProject\DataMain.xlsx',engine='openpyxl')

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

for i in range(0 , rows_student):
    dataMain.at[i , 'id_student'] = X_student[i][0]
    for j in range(0,len(activity_type)):
        sum = 0
        flag = False
        for k in range(0,len(X_studentVle)):
            if(X_student[i][0]==X_studentVle[k][0] and activity_type[j]==X_studentVle[k][2]):
                sum = sum + int(X_studentVle[k][1])
                X_studentVle = np.delete(X_studentVle , k)
                flag = True
            if(X_student[i][0] != X_studentVle[k][0] and flag == True):
                break
            
        dataMain.at[i , activity_type[j]] = sum
    
    if i==1000 :
        len(X_studentVle)
    dataMain.at[i , 'final_result'] = X_student[i][1]
end = time.time()

print(end-start)
dataMain.to_excel(r'C:\Users\hp\Desktop\FinalProject\DataMain.xlsx', index=False)
