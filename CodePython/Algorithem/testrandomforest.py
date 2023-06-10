import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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

X=df.drop(['final_result'] , axis=1)

y=df['final_result']

#split data into separate training and test set

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)   # I changed 0.3 to 0.2

print("# check the shape of X_train")
print(X_train.shape)

print("# check the shape of X_test")
print(X_test.shape)

#feature engineering

print("\n=================== Feature engineering ==================\n\n")

print("# check data types in X_train")

print(X_train.dtypes)

print(X_train.isnull().sum())

print("=======================================================\n\n")

X_train.head()

X_train.shape

X_test.head()

X_test.shape

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

print(y_test.describe())

# check null accuracy score

null_accuracy = (y_test.describe()[3]/y_test.describe()[0])

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

print("============ Confusion matrix ==============")

print("# Print the Confusion Matrix and slice it into four pieces")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=rfc.classes_)

print('Confusion matrix\n\n', cm)

# print('\nTrue Positives(TP) = ', cm[0,0])

# print('\nTrue Negatives(TN) = ', cm[1,1])

# print('\nFalse Positives(FP) = ', cm[2,1])

# print('\nFalse Negatives(FN) = ', cm[1,0])

print("# visualize confusion matrix with seaborn heatmap")

cm_matrix = pd.DataFrame(data=cm)

sns.heatmap(cm_matrix, annot=True, fmt='d', xticklabels=rfc.classes_ , yticklabels=rfc.classes_)

plt.ylabel('prediction', fontsize=13)
plt.xlabel('actual', fontsize=13)
plt.title('confusion matrix', fontsize=17)
plt.show()


print("================= Classification metrices ===============")

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

#======Distinction======

TPD = cm[0][0]
TND = cm[1][1] + cm[1][2] + cm[1][3] + cm[2][1] + cm[2][2] + cm[3][3] + cm[3][1] + cm[3][2] + cm[3][3]    
FPD = cm[0][1] + cm[0][2] + cm[0][3]
FND = cm[1][0] + cm[2][0] + cm[3][0]


#======Fail=======

TPF = cm[1][1]
TNF = cm[0][0] + cm[0][2] + cm[0][3] + cm[2][0] + cm[2][2] + cm[2][3] + cm[3][0] + cm[3][2] + cm[3][3]    
FPF = cm[1][1] + cm[1][2] + cm[1][3]
FNF = cm[0][1] + cm[2][1] + cm[3][1]

#======Pass=======

TPP = cm[2][2]
TNP = cm[0][0] + cm[0][1] + cm[0][3] + cm[1][0] + cm[1][1] + cm[1][3] + cm[3][0] + cm[3][1] + cm[3][3]    
FPP = cm[2][0] + cm[2][1] + cm[2][3]
FNP = cm[0][2] + cm[1][2] + cm[3][2]

#======Withdrawn=======

TPW = cm[3][3]
TNW = cm[0][0] + cm[0][1] + cm[0][2] + cm[1][0] + cm[1][1] + cm[1][2] + cm[2][0] + cm[2][1] + cm[2][2]    
FPW = cm[3][0] + cm[3][1] + cm[3][2]
FNW = cm[0][3] + cm[1][3] + cm[2][3]

print("\n print'Distinction' classification accuracy\n")

classification_accuracy_D = (TPD + TND) / float(TPD + TND + FPD + FND)

print('Distinction Classification accuracy : {0:0.4f}'.format(classification_accuracy_D))

print("# print Distinction classification error")

classification_error_D = (FPD + FND) / float(TPD + TND + FPD + FND)

print('Distinction Classification error : {0:0.4f}'.format(classification_error_D))

print("# print Distinction precision score")

precisionD = TPD / float(TPD + FPD)


print('Distinction Precision : {0:0.4f}'.format(precisionD))

#================

recallD = TPD / float(TPD + FND)

print('Distinction Recall or Sensitivity : {0:0.4f}'.format(recallD))

#================
true_positive_rate_D= TPD / float(TPD + FND)

print('Distinction True Positive Rate : {0:0.4f}'.format(true_positive_rate_D))

#==============
false_positive_rate_D = FPD / float(FPD + TND)

print('Distinction False Positive Rate : {0:0.4f}'.format(false_positive_rate_D))

#===============Fail=============


print("\n print'Fail' classification accuracy\n")

classification_accuracy_F = (TPF + TNF) / float(TPF + TNF + FPF + FNF)

print('Fail Classification accuracy : {0:0.4f}'.format(classification_accuracy_F))

print("# print Fail classification error")

classification_error_F = (FPF + FNF) / float(TPF + TNF + FPF + FNF)

print('Fail Classification error : {0:0.4f}'.format(classification_error_F))

print("# print Fail precision score")

precisionF = TPF / float(TPF + FPF)


print('Fail Precision : {0:0.4f}'.format(precisionF))

#================

recallF = TPF / float(TPF + FNF)

print('Fail Recall or Sensitivity : {0:0.4f}'.format(recallF))

#================
true_positive_rate_F= TPF / float(TPF + FNF)

print('Fail True Positive Rate : {0:0.4f}'.format(true_positive_rate_F))

#==============
false_positive_rate_F = FPF / float(FPF + TNF)

print('Fail False Positive Rate : {0:0.4f}'.format(false_positive_rate_F))

#===============Pass=============


print("\n print'Pass' classification accuracy\n")

classification_accuracy_P = (TPP + TNP) / float(TPP + TNP + FPP + FNP)

print('Pass Classification accuracy : {0:0.4f}'.format(classification_accuracy_P))

print("\n print Pass classification error\n")

classification_error_P = (FPP + FNP) / float(TPP + TNP + FPP + FNP)

print('Pass Classification error : {0:0.4f}'.format(classification_error_P))

print("# print Pass precision score")

precisionP = TPP / float(TPP + FPP)


print('Pass Precision : {0:0.4f}'.format(precisionP))

#================

recallP = TPP / float(TPP + FNP)

print('Pass Recall or Sensitivity : {0:0.4f}'.format(recallP))

#================
true_positive_rate_P= TPP / float(TPP + FNP)

print('Pass True Positive Rate : {0:0.4f}'.format(true_positive_rate_P))

#==============
false_positive_rate_P = FPP / float(FPP + TNP)

print('Pass False Positive Rate : {0:0.4f}'.format(false_positive_rate_P))

#===============Whitdrawn=============


print("\n print'Whitdrawn' classification accuracy\n")

classification_accuracy_W = (TPW + TNW) / float(TPW + TNW + FPW + FNW)

print('Whitdrawn Classification accuracy : {0:0.4f}'.format(classification_accuracy_W))

print("# print Whitdrawn classification error")

classification_error_W = (FPW + FNW) / float(TPW + TNW + FPW + FNW)

print('Whitdrawn Classification error : {0:0.4f}'.format(classification_error_W))

print("# print Whitdrawn precision score")

precisionW = TPW / float(TPW + FPW)


print('Whitdrawn Precision : {0:0.4f}'.format(precisionW))

#================

recallW = TPW / float(TPW + FNW)

print('Whitdrawn Recall or Sensitivity : {0:0.4f}'.format(recallW))

#================
true_positive_rate_W= TPW / float(TPW + FNW)

print('Whitdrawn True Positive Rate : {0:0.4f}'.format(true_positive_rate_W))

#==============
false_positive_rate_W = FPW / float(FPW + TNW)

print('Whitdrawn False Positive Rate : {0:0.4f}'.format(false_positive_rate_W))



print("==================== Calculate class probabilities ==================")

print("# print the first 10 predicted probabilities of classes 'Distinction' 'Fail' 'Pass' 'Withdrawn' ")


y_pred_prob = rfc.predict_proba(X_test)[0:10]

print(y_pred_prob)

print("# store the probabilities in dataframe")
#==============?===============
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=rfc.classes_)

print(y_pred_prob_df)

print("# print the first 10 predicted probabilities for class 'Distinction' ")

print(rfc.predict_proba(X_test)[0:10, 0])

print("# print the first 10 predicted probabilities for class 'Fail'")

print(rfc.predict_proba(X_test)[0:10, 1])

print("# print the first 10 predicted probabilities for class 'Pass' ")

print(rfc.predict_proba(X_test)[0:10, 2])

print("# print the first 10 predicted probabilities for class ' Withdrawn ' ")

print(rfc.predict_proba(X_test)[0:10, 3])

# store the predicted probabilities for class Fail 

print("plot histogram for class Distinction ")

y_pred0 = rfc.predict_proba(X_test)[:, 0]

print(y_pred0)

# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12

# plot histogram with 10 bins
plt.hist(y_pred0, bins = 10)
# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of Distinction')
# set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of Distinction')
plt.ylabel('Frequency')
plt.show()

print("plot histogram for class 'Fail' ")

y_pred1 = rfc.predict_proba(X_test)[:, 1]

print(y_pred1)

# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12

# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)
# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of Fail')
# set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of Fail')
plt.ylabel('Frequency')
plt.show()

print("plot histogram for class Pass")

y_pred2 = rfc.predict_proba(X_test)[:, 2]

print(y_pred2)

# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12

# plot histogram with 10 bins
plt.hist(y_pred2, bins = 10)
# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of Pass')
# set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of Pass')
plt.ylabel('Frequency')
plt.show()

print("plot histogram for class Withdrawn  ")

y_pred3 = rfc.predict_proba(X_test)[:, 3]

print(y_pred3)

# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12

# plot histogram with 10 bins
plt.hist(y_pred3, bins = 10)
# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of Withdrawn ')
# set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of Withdrawn ')
plt.ylabel('Frequency')
plt.show()