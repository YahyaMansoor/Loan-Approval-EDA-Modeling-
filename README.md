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

For loan approval prediction, we should use supervised learning, because we have a clear target variable (approval or denial) and labeled data, so supervised learning is the appropriate approach. You can use techniques such as logistic regression, decision trees, or neural networks to build a predictive model that can accurately classify loan applications as approved or denied based on their input features.

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
