import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


import os
for dirname, _, filenames in os.walk(r'C:\Users\sajad\Desktop\FinalProject\data.xlsx'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')

#impory dataset

# I consider that we have 100% of data


df=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\data.xlsx',engine='openpyxl')


#I consider that there is no need to exploratory analysis
# steps of exploratory: 
# 1- rename columns name 
# 2- explore categorical var
# 3- explor and fix problems of categorical var (missing values, frequency count , encoded values, check missing values again,cardinality)
# 4- explore numerical var
# 5- explor and fix problems of numericall var (missing values,...)

# declare feature vector and target

#----------- ? ------------


X=df.drop(['final_result'] , axis=1)

y=df['final_result']

#split data into separate training and test set

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)   # I changed 0.3 to 0.2

# check the shape of X_train and X_test

X_train.shape, X_test.shape

#feature engineering
#I consider there is no missing values and no need to feature engineering 
# feature scaling

cols = X_train.columns
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.head()

#We now have X_train dataset ready to be fed into the Gaussian Naive Bayes classifier. I will do it as follows.

#model training
#=================
# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# instantiate the model
gnb = GaussianNB()
rfc = RandomForestClassifier(n_estimators=2000)


# fit the model
# gnb.fit(X_train, y_train)

rfc.fit(X_train, y_train)

#predct the result
#=================

# y_pred = gnb.predict(X_test)

y_pred = rfc.predict(X_test)

#check accuracy score
#====================


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#Compare the train-set and test-set accuracy

y_pred_train = rfc.predict(X_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

#Check for overfitting and underfitting

# print the scores on training and test set

print('Training set score: {:.4f}'.format(rfc.score(X_train, y_train)))

#Check for overfitting and underfitting¶
# print the scores on training and test set

print('Training set score: {:.4f}'.format(rfc.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(rfc.score(X_test, y_test)))

#Compare model accuracy with null accuracy

# check class distribution in test set

y_test.value_counts()

# check null accuracy score

null_accuracy = (7407/(7407+2362))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
