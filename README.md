# Loan-Approval-EDA-Modeling-


## Problem Scoping

The loan approval prediction can be used by banks or other financial institutions to help them make more accurate and efficient loan approval decisions. It can also be used by individuals who are seeking loans to understand their chances of approval and what factors are considered in the decision-making process.

The loan approval prediction it's important because it can improve the loan approval process by making it more efficient, reducing the time and resources required for manual underwriting, and providing more accurate and consistent decisions. It can also reduce bias and discrimination by using objective data-driven models to make decisions.

The social impact of The loan approval prediction that can help increase financial inclusion by enabling more people to access loans and credit, particularly those who may have been overlooked or discriminated against in the past. It can also help reduce economic inequality by providing fairer access to credit.

There are several ethical considerations to keep in mind when developing a loan approval prediction. One is the potential for bias and discrimination in the data and models used, which could lead to unfair treatment of certain groups of people. It's important to ensure that the data used is representative of the population and that the models are designed to minimize bias. Another consideration is transparency and accountability - individuals should be able to understand why they were approved or denied for a loan, and there should be a mechanism for challenging decisions if they are deemed unfair or discriminatory. Finally, privacy is also a concern, as loan application data may contain sensitive personal information that needs to be protected.


How to convert loan approval prediction problem statement to AI problem statement ? 
To convert the loan approval prediction problem statement to an AI problem statement, we need to define the task as a machine learning problem that uses historical data to predict whether a new loan application should be approved or denied.

Here are some steps you can follow to convert the problem statement:

1. Collect historical loan data: The first step is to collect data on past loan applications, including information on the applicant's credit history, income, debt-to-income ratio, loan amount, loan purpose, and whether the loan was approved or denied.

2. Define the problem statement: The loan approval prediction problem statement can be defined as a binary classification task, where the goal is to predict whether a new loan application will be approved or denied based on the input features.

3. Feature engineering: Next, we need to preprocess and engineer the input features to prepare them for machine learning algorithms. This may involve data cleaning, feature scaling, encoding categorical variables, and handling missing data.

4. Select a machine learning algorithm: We can then select an appropriate machine learning algorithm that is suitable for the loan approval prediction problem, such as logistic regression, decision trees, or neural networks.

5. Train the model: We can use the historical loan data to train the machine learning model, using techniques such as cross-validation to evaluate its performance.

6. Evaluate the model: Once the model is trained, we can evaluate its performance on a test set of loan applications that were not used for training. This can involve metrics such as accuracy, precision, recall, and F1 score.

7. Deploy the model: Finally, we can deploy the trained model in a production environment, where it can be used to make predictions on new loan applications in real-time. We may also need to monitor the model's performance over time and update it as new data becomes available.


What should I use ? Supervised Learning or Unsuperveised Learning?

For loan approval prediction, we should use supervised learning, because we have a clear target variable (approval or denial) and labeled data, so supervised learning is the appropriate approach. You can use techniques such as Random Forest Classifier, Decision Trees, K- Nearest Neighbours Classifiers, Linear SVM (Support Vector Machines) Classifier, or Logistic Regression to build a predictive model that can accurately classify loan applications as approved or denied based on their input features.

## Data Acquistion

``` python

#Read CSV data
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Projects/(Loan Prediction).csv")
#preview data
data.head()

```

Features:

Gender: Male/Female (Basic Information)

Married: Yes/No (This is noted to check the dependecy as well as the liabilities of the candidate.)

Dependents: no.of dependents (This is again recorded to check the liabilities of the candidate.)

Education: graduate/Not graduate (This helps the lender understand the loan repayment capacity of the candidate.)

Self Employment: Yes/No (This keeps a check on the flow of income of the borrower)

Co-appliant income:number (This helps to understand the loan repayment capacity of the borrower.)

Loan Amount: number (This helps the lender understand the feasibility of the repayment of the borrower.)

Loan Amount term: number (Helps the lender to calculate the the total interest with the principal amount.)

Property Area: Uraban/Rural/Semi-Urban (Helps the lender understand the standard of living of the borrower.)

Loan Status: Yes/No (This is the final decision made by the lender)
## Data Exlploration
Over here we are trying to find the missing data from our dataset.
Missing data could be for many reasons like :

1.Test design

2.failure in observation

3.failure to record observation

Correcting Missing data is very crucial because:

1.It can affect the decisions of the ML models, which can reduce the accuracy of the model.

2.It makes the model baised to the one class,where all the datapoints are present.This also leads to inaccuracy which cannot be recorded.

It is always better to check the number of missing data:

1.If the missing data is more in number, then the possible solution is to impute the missing places by taking the mean of the row/column.

2.If the missing data is less in number , then the missing row can be deleted.

``` python
#Preview data information
data.info()
#Check missing values
data.isnull().sum()
```

After figuring out the missing values of all the feature/columns we can now impute or detele the missing values.
Criteria for imputing data:

1. If the feature is categorical in nature (e.g Temperature can be hot,humid,cold) we can impute it with the most occuring category.

2. If the feautue is numerical in nature (e.g Height of the person) we can impute the missing row by taking the mean or median of the enitre feature.

Caution before deleting the missing data/rows:
We should only delete the missing data when the number of missing rows are less in number. Deleting rows can solve missing data problem, but it is also loss of data when it comes to other features in that row.

## Final Adjustments to Data

Based on my assessment of the missing values in the dataset, I'll make the following changes to the data:

If "Gender" is missing for a given row, I'll impute with Male (most common answer).
If "Married" is missing for a given row, I'll impute with yes (most common answer).
If "Dependents" is missing for a given row, I'll impute with 0 (most common answer).
If "Self_Employed" is missing for a given row, I'll impute with no (most common answer).
If "LoanAmount" is missing for a given row, I'll impute with mean of data.
If "Loan_Amount_Term" is missing for a given row, I'll impute with 360 (most common answer).
If "Credit_History" is missing for a given row, I'll impute with 1.0 (most common answer).

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
