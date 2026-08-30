# Data Dictionary

## Static Attributes (`loan_static_attributes.csv`)
* `loan_id`: Unique loan identifier (string).
* `orig_balance`: Original loan balance at origination (numeric).
* `credit_score`: Borrower credit score at origination (numeric).
* `ltv`: Loan-to-Value ratio at origination (numeric).
* `dti`: Debt-to-Income ratio at origination (numeric).
* `state`: US State code (string).
* `loan_purpose`: Purpose of loan (P=Purchase, C=Cash-out Refi, R=Rate/Term Refi).
* `property_type`: Type of property (SF=Single Family, CO=Condo, PU=PUD).
* `vintage`: Year/Quarter of origination (string).

## Monthly Performance (`loan_monthly_performance_*.csv`)
* `loan_id`: Unique loan identifier.
* `reporting_period`: Month and year of the record (MMYYYY).
* `current_upb`: Current unpaid principal balance.
* `current_interest_rate`: Current interest rate.
* `loan_age`: Age of the loan in months.
* `remaining_months`: Remaining months to maturity.
* `delinquency_status`: Current delinquency status (0=Current, 1=30D, 2=60D, etc.).
* `zero_balance_code`: Reason for zero balance (e.g., 01=Prepaid, 02=Third Party Sale, 03=Short Sale, 09=REO).

## Servicer Updates (`servicer_updates.csv`)
* `loan_id`: Unique loan identifier.
* `reporting_period`: Month and year of the record (MMYYYY).
* `delinquency_status`: Updated delinquency status.
* `current_upb`: Updated current unpaid principal balance.
* `update_date`: Date the servicer update was received.
