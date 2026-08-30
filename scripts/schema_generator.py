import json
import csv
import os

def create_data_dictionary():
    with open("data_dictionary.md", "w") as f:
        f.write("""# Data Dictionary

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
""")

def create_validation_rules():
    rules = {
        "balance_consistency": {
            "description": "Current UPB cannot be greater than original balance + tolerance.",
            "type": "inequality",
            "check": "current_upb <= orig_balance * 1.05"
        },
        "delinquency_progression": {
            "description": "Delinquency status cannot jump more than 1 month at a time.",
            "type": "sequential",
            "check": "delinquency_status(t) <= delinquency_status(t-1) + 1"
        },
        "closed_prepaid_status": {
            "description": "If UPB is 0, zero_balance_code must be populated.",
            "type": "conditional",
            "check": "if current_upb == 0 then zero_balance_code is not null"
        }
    }
    with open("validation_rules.json", "w") as f:
        json.dump(rules, f, indent=4)

def create_macro_scenarios():
    scenarios = [
        {"scenario_id": "Base", "unemployment_rate": 0.04, "hpi_growth": 0.03, "interest_rate": 0.06},
        {"scenario_id": "Adverse", "unemployment_rate": 0.08, "hpi_growth": -0.05, "interest_rate": 0.07},
        {"scenario_id": "High-Prepay", "unemployment_rate": 0.03, "hpi_growth": 0.06, "interest_rate": 0.04},
    ]
    with open("macro_scenarios.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=scenarios[0].keys())
        writer.writeheader()
        writer.writerows(scenarios)

def create_submission_template():
    template = [
        {"loan_id": "EXAMPLE1", "reporting_period": "012026", "prob_default": 0.05, "prob_prepay": 0.15, "next_state": "Current", "exception_type": "None", "anomaly_score": 0.01, "top_drivers": "credit_score, ltv", "action": "Accept", "confidence": 0.95}
    ]
    with open("submission_template.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=template[0].keys())
        writer.writeheader()
        writer.writerows(template)

if __name__ == "__main__":
    create_data_dictionary()
    create_validation_rules()
    create_macro_scenarios()
    create_submission_template()
    print("Schema files generated successfully.")
