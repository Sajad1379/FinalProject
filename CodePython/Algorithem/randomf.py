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

print("============ Exploratory data analysis ============")
print("# view dimensions of dataset")

print(df.shape)

print("# preview the dataset")

print(df.head())

print("#Rename column names: but we don't need")

print("# view summary of dataset")

print(df.info())

print("# find categorical variables")

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

print("# view the categorical variables")

print(df[categorical].head())

print("# check missing values in categorical variabl")

print(df[categorical].isnull().sum())

print("#if we have missing values ")

print("# view frequency counts of values in categorical variables")

for var in categorical: 
    
    print(df[var].value_counts())

print("# view frequency distribution of categorical variables")

for var in categorical: 
    
    print(df[var].value_counts()/float(len(df)))

print("#the outputs like example(in site) shows missing values: i name them missing 1 , 2 , 3 ,...")

print("# check labels in missing1(2 , 3 ,...) variable")

print(df.missing1.unique())

print("# check frequency distribution of values in missing1(2 , 3 ,...) variable")

print(df.missing1.value_counts())

print("# replace 'inappropriate' values in missing1(2 , 3 ,...) variable with `NaN`")

df['missing1'].replace('inappropriate', np.NaN, inplace=True)

print("#again check the frequency distribution of values in missing1(2 , 3 ,...) variable")

print(df.missing1.value_counts())

print("# check for cardinality in categorical variables")

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')

print("Explore Numerical Variables")

print("find numerical variables")


numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))


print('The numerical variables are :', numerical)


print("view the numerical variables")


print(df[numerical].head())

print("==================================================================")

# declare feature vector and target

#----------- ? ------------


X=df.drop(['final_result'] , axis=1)

y=df['final_result']

#split data into separate training and test set

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)   # I changed 0.3 to 0.2

print("# check the shape of X_train and X_test")

print(X_train.shape)
print(X_test.shape)

#feature engineering

print("\n=================== Feature engineering ==================\n\n")


print("# check data types in X_train")

print(X_train.dtypes)

print("# display categorical variables")

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

print(categorical)

print("# display numerical variables")

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

print(numerical)

print("# print percentage of missing values in the categorical variables in training set")

print(X_train[categorical].isnull().mean())

print("# print categorical variables with missing data")

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))

print("# impute missing categorical variables with most frequent value")

for df2 in [X_train, X_test]:
    df2['missing1'].fillna(X_train['missing1'].mode()[0], inplace=True)
    df2['missing12,3,...'].fillna(X_train['missing2,3x,...'].mode()[0], inplace=True)
 
print("# check missing values in categorical variables in X_train")

print(X_train[categorical].isnull().sum())

print("# check missing values in X_train")

print(X_train.isnull().sum())

print("# check missing values in X_test")

print(X_test.isnull().sum())

print("# Encode categorical variables \n\n print categorical variables")

print(categorical)

print(X_train[categorical].head())

print("# import category encoders")

import category_encoders as ce

print("# encode remaining variables with one-hot encoding")

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
                                 'race', 'sex', 'native_country'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

print(X_train.head())

print(X_train.shape)

print(X_test.head())

print(X_test.shape)

print("=======================================================\n\n")

print("# feature scaling")
cols = X_train.columns
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
print(X_train.head())

#We now have X_train dataset ready to be fed into the Gaussian Naive Bayes classifier. I will do it as follows.

print("#model training")
#=================

# train a Gaussian Naive Bayes classifier on the training set
#naive bayes #from sklearn.naive_bayes import GaussianNB
#random forest
from sklearn.ensemble import RandomForestClassifier

#random forest
rfc = RandomForestClassifier(n_estimators=2000)


print("# fit the model")
print(rfc.fit(X_train, y_train))

print("#predct the result")
#=================

y_pred = rfc.predict(X_test)

print(y_pred)

print("================ check accuracy score ================")
#====================


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

print("#Compare the train-set and test-set accuracy")

y_pred_train = rfc.predict(X_train)

print(y_pred_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print("Check for overfitting and underfitting")

# print the scores on training and test set

print('Training set score: {:.4f}'.format(rfc.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(rfc.score(X_test, y_test)))

print("Compare model accuracy with null accuracy")

# check class distribution in test set

print(y_test.value_counts())

# check null accuracy score

null_accuracy = (7407/(7407+2362))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

print("============ Confusion matrix ==============")

print("# Print the Confusion Matrix and slice it into four pieces")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

print("# visualize confusion matrix with seaborn heatmap")

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

print(sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Ylrfcu'))

print("================= Classification metrices ===============")

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

print("\n print classification accuracy")

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

print("# print classification error")

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

print("# print precision score")

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))

#================

recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))

#================
true_positive_rate = TP / float(TP + FN)

print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

#==============
false_positive_rate = FP / float(FP + TN)

print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

print("==================== Calculate class probabilities ==================")

print("# print the first 10 predicted probabilities of two classes- 0 and 1")

y_pred_prob = rfc.predict_proba(X_test)[0:10]

print(y_pred_prob)

print("# store the probabilities in dataframe")

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['1', '2', '3', '4'])

print(y_pred_prob_df)

print("# print the first 10 predicted probabilities for class 1 - Probability 1")

print(rfc.predict_proba(X_test)[0:10, 1])

# store the predicted probabilities for class 1 - Probability of 1

y_pred1 = rfc.predict_proba(X_test)[:, 1]


# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of 1')


# set the x-axis limit
plt.xlim(0,1)

# set the title
plt.xlabel('Predicted probabilities of 1')
plt.ylabel('Frequency')
#=============== ? ==================
print(plt.xlabel('Predicted probabilities of 1'))
print(plt.ylabel('Frequency'))

print("==================== ROC - AUC ========================")

print("# plot ROC Curve")

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = '>50K')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting finalresult')

#=========== ? ==============

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

print("# compute ROC AUC")

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))

print("# calculate cross-validated ROC AUC ")

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(rfc, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))


print("================ k-Fold Cross Validation ====================")

print("# Applying 10-Fold Cross Validation")

from sklearn.model_selection import cross_val_score

scores = cross_val_score(rfc, X_train, y_train, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))

print("# compute Average cross-validation score")

print('Average cross-validation score: {:.4f}'.format(scores.mean()))










