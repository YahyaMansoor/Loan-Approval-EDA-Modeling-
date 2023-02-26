# Loan-Approval-EDA-Modeling-


## Problem Scoping



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
