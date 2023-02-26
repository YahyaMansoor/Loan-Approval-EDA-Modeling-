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


