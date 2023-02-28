# Loan-Approval-EDA-Modeling-

# Table of Contents
1. [Problem Scoping](#Problem-Scoping)
2. [Import Packages](#Import-Packages)
3. [Data Acquistion](#Data-Acquistion)
4. [Data Exploration / EDA (Exploratory Data Analysis)](#Data-Exploration-/-EDA-(Exploratory-Data-Analysis))
5. [Final Adjustments to Data](#Final-Adjustments-to-Data)
6. [Modeling](#Modeling)

## Problem Scoping

The loan approval prediction is important for improving the loan approval process and increasing financial inclusion. It can be converted into an AI problem statement using supervised learning techniques such as Random Forest Classifier, Decision Trees, K- Nearest Neighbours Classifiers, or Logistic Regression. Historical loan data is collected and preprocessed, and a machine learning algorithm is selected and trained and the model's performance is evaluated. Supervised learning is the appropriate approach because we have a clear target variable and labeled data.

## Import Packages

``` python

#Import packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
```

## Data Acquistion

``` python

#Read CSV data
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Projects/(Loan Prediction).csv")
#preview data
data.head()

```

Features:

- Gender: Male/Female (Basic Information)

- Married: Yes/No (This is noted to check the dependecy as well as the liabilities of the candidate.)

- Dependents: no.of dependents (This is again recorded to check the liabilities of the candidate.)

- Education: graduate/Not graduate (This helps the lender understand the loan repayment capacity of the candidate.)

- Self Employment: Yes/No (This keeps a check on the flow of income of the borrower)

- Co-appliant income:number (This helps to understand the loan repayment capacity of the borrower.)

- Loan Amount: number (This helps the lender understand the feasibility of the repayment of the borrower.)

- Loan Amount term: number (Helps the lender to calculate the the total interest with the principal amount.)

- Property Area: Uraban/Rural/Semi-Urban (Helps the lender understand the standard of living of the borrower.)

- Loan Status: Yes/No (This is the final decision made by the lender)

## Data Exploration / EDA (Exploratory Data Analysis)

Over here we are trying to find the missing data from our dataset.
- Missing data can occur due to various reasons and correcting it is crucial for accurate machine learning models. 
- Missing data can lead to biased results and reduced accuracy. 
- It's important to check the number of missing data and take appropriate actions such as imputing the missing values or deleting the missing rows/columns depending on the number of missing data.

``` python
#Preview data information
data.info()
#Check missing values
data.isnull().sum()
```

After identifying the missing values in the dataset, we can either impute or delete them. If the missing feature is categorical, we can impute it with the most occurring category. On the other hand, if the missing feature is numerical, we can impute it with the mean or median of the entire feature. However, caution should be taken before deleting missing data, especially when the number of missing rows is less. Deleting rows may solve the missing data problem, but it also leads to the loss of data from other features in that row.


## Final Adjustments to Data

Based on my assessment of the missing values in the dataset, I'll make the following changes to the data:

- If "Gender" is missing for a given row, I'll impute with Male (most common answer).
- If "Married" is missing for a given row, I'll impute with yes (most common answer).
- If "Dependents" is missing for a given row, I'll impute with 0 (most common answer).
- If "Self_Employed" is missing for a given row, I'll impute with no (most common answer).
- If "LoanAmount" is missing for a given row, I'll impute with mean of data.
- If "Loan_Amount_Term" is missing for a given row, I'll impute with 360 (most common answer).
- If "Credit_History" is missing for a given row, I'll impute with 1.0 (most common answer).

``` python
train_data = data.copy()
train_data['Gender'].fillna(train_data['Gender'].value_counts().idxmax(), inplace=True)
train_data['Married'].fillna(train_data['Married'].value_counts().idxmax(), inplace=True)
train_data['Dependents'].fillna(train_data['Dependents'].value_counts().idxmax(), inplace=True)
train_data['Self_Employed'].fillna(train_data['Self_Employed'].value_counts().idxmax(), inplace=True)
train_data["LoanAmount"].fillna(train_data["LoanAmount"].mean(skipna=True), inplace=True)
train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
train_data['Credit_History'].fillna(train_data['Credit_History'].value_counts().idxmax(), inplace=True)
```

Converting categorical data into numerical data.

In the following cell , we will be converting categorical data to numerical data, as ML models can only take in numbers and perform operations on them. Any data irrespective of the form(picture,text) has to be converted to numerial form before training the machine learning model on it.
For example the feature 'gender' has two categories 'Male' and 'Female' which is being converted to 1 and 0 respectively.
The feature 'Property Area' has three categories 'Semi-Urban', 'Urban' and 'Rural' which is being converted to 0,1 and 2 respectively.

``` python
#Convert some object data type to int64
gender_stat = {"Female": 0, "Male": 1}
yes_no_stat = {'No' : 0,'Yes' : 1}
dependents_stat = {'0':0,'1':1,'2':2,'3+':3}
education_stat = {'Not Graduate' : 0, 'Graduate' : 1}
property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}
Y_N_stat = {'N' : 0,'Y' : 1}

train_data['Gender'] = train_data['Gender'].replace(gender_stat)
train_data['Married'] = train_data['Married'].replace(yes_no_stat)
train_data['Dependents'] = train_data['Dependents'].replace(dependents_stat)
train_data['Education'] = train_data['Education'].replace(education_stat)
train_data['Self_Employed'] = train_data['Self_Employed'].replace(yes_no_stat)
train_data['Property_Area'] = train_data['Property_Area'].replace(property_stat)
train_data['Loan_Status'] = train_data['Loan_Status'].replace(Y_N_stat)
``` 

## Modeling

Handling Data Imbalance with SMOTE and splitting the data.
In the following cell we will balance the data using Oversampling technique - SMOTE and Split the data to test and train data.

``` python

# Split the Data into the train and test sets.
x = train_data.drop(['Loan_Status','Loan_ID'],axis=1)   # Data consisting of other features 
y = train_data['Loan_Status'] # Data containing the Loan_Status variable
# Oversampling technique- SMOTE
sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(x, y)

# Split the Data into the train and test sets such that test set has 30% of the values.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
``` 

Now I will start build and evaluate the models.

1. Random Forest Classifier

``` python
# using GridSearchCV to find the best parameters.
# RandomForest
rf_params = {'n_estimators':[100,200,300],'max_features':['auto','sqrt'],"min_samples_leaf":[1,2,4],"bootstrap":[True, False]}
grid_rf = GridSearchCV(RandomForestClassifier(),rf_params)
grid_rf.fit(x_train, y_train)
rf = grid_rf.best_params_
print(rf)
``` 


``` python
# Build the Random Forest Classifier prediction model.
rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(x_train,y_train)
#Evaluation of Random Forest Classifier 
rf_y_pred = rf_clf.predict(x_test)
print("Confusion Matrix")
print(confusion_matrix(y_test, rf_y_pred))
print("*"*100)
print("Classification report")
print(classification_report(y_test, rf_y_pred))
```

2. Decision Tree Classifier

``` python
# using GridSearchCV to find the best parameters.
# Decision Tree
tree_params = {'criterion':['gini','entropy'],"max_depth":[2,4,6],"min_samples_leaf":[5,7,9,]}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(x_train, y_train)
dcTree = grid_tree.best_params_
print(dcTree)
``` 


``` python
# Build the Decision Tree Classifier prediction model.
clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)
# Evaluation of Decision Tree Classifier
y_pred = clf.predict(x_test)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("*"*100)
print("Classification report")
print(classification_report(y_test, y_pred))
```
3. K- Nearest Neighbours Classifier

``` python
# using GridSearchCV to find the best parameters.
# KNN
knn_params = {"n_neighbors":[1,3,5], 'algorithm':['auto']}
grid_knn = GridSearchCV(KNeighborsClassifier(),knn_params)
grid_knn.fit(x_train, y_train)
knn = grid_knn.best_params_
print(knn)
``` 

``` python
# Build the K- Nearest Neighbours Classifier prediction model.
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train,y_train)
# Evaluation of K- Nearest Neighbours Classifier
knn_y_pred = knn_clf.predict(x_test)
print("Confusion Matrix")
print(confusion_matrix(y_test, knn_y_pred))
print("*"*100)
print("Classification report")
print(classification_report(y_test, knn_y_pred))
``` 
4. Linear SVM Classifier
``` python
# Build the Linear SVM Classifier prediction model.
svm_clf  =  svm.LinearSVC(max_iter=5000)
svm_clf.fit(x_train,y_train)
# Evaluation of Linear SVM Classifier
svm_y_pred = svm_clf.predict(x_test)
print("Confusion Matrix")
print(confusion_matrix(y_test, svm_y_pred))
print("*"*100)
print("Classification report")
print(classification_report(y_test, svm_y_pred))
``` 
5. Logistic Regression

``` python
# using GridSearchCV to find the best parameters.
# Logistic Regression
log_reg_params = {"penalty":['l1','l2'], 'C':[0.001, 0.01, 0.1,1,10,100],'solver':['liblinear']}
grid_log_reg = GridSearchCV(LogisticRegression(),log_reg_params)
grid_log_reg.fit(x_train, y_train)
log_reg = grid_log_reg.best_params_
print(log_reg)
```


``` python
# Build the Logistic Regression model.
log_reg = LogisticRegression(n_jobs = -1)
log_reg.fit(x_train, y_train)
log_reg.score(x_train, y_train)
# Evaluation of Logistic Regression model
y_pred = log_reg.predict(x_test)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("*"*100)
print("Classification report")
print(classification_report(y_test, y_pred))
``` 
