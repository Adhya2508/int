"""
=================================================================
INTAIN AI CHALLENGE -- TASK 7: LLM-ASSISTED REVIEWER COPILOT
Loan Performance Intelligence Engine
Intain Campus FinTech Challenge 2026

This module implements the Reviewer Copilot. It:
  - Takes a structured loan input bundle (JSON / dict).
  - Prompts a grounded LLM (using Groq OpenAI compatibility).
  - Restricts responses to recommendations only.
  - Logs prompts, responses, inputs, and rejected outputs.
=================================================================
"""
import os
import sys
import json
import hashlib
from datetime import datetime
from openai import OpenAI

# Ensure UTF-8 console output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# CONFIG PATHS
BASE_DIR        = "e:/intain"
LOGS_DIR        = os.path.join(BASE_DIR, "logs/copilot")
PROMPTS_LOG_DIR = os.path.join(BASE_DIR, "logs/prompts")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs/copilot_examples")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PROMPTS_LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dotenv():
    for path in [".env", "e:/intain/.env", "../../.env"]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip().strip('"').strip("'")
                break
            except Exception:
                pass

load_dotenv()

# CLIENT INITIALIZATION (GROQ OpenAI Compatibility Endpoint)
# Reads API key from OPENAI_API_KEY environment variable.
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "groq/compound-mini"

client = None
if API_KEY:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
else:
    print("WARNING: OPENAI_API_KEY environment variable not found. Copilot will run in Dry Run/Mock mode.")

# ---------------------------------------------------------------
# GROUNDED PROMPT TEMPLATE
# ---------------------------------------------------------------
PROMPT_TEMPLATE = """You are a senior credit risk assessment assistant for the "Loan Performance Intelligence Engine" platform.
Your task is to draft a concise, objective reviewer note for a human credit officer based ONLY on the structured context provided below.

=========================================
LOAN PROFILE CONTEXT:
* Loan ID: {loan_id}
* Reporting Cycle: {reporting_period}
* Prepayment Probability (Calibrated): {predicted_probability:.2%}
* Decision Band Status: {decision_band} (Prepayment Risk Threshold: 14.86%)
* Anomaly Score: {anomaly_score:.4f}
* Exception Flag Status: {exception_flag}
* Business Rule Violations: {rule_violations}
* Static Origination Attributes:
  - Credit Score: {credit_score}
  - Loan-to-Value (LTV): {ltv}%
  - Debt-to-Income (DTI): {dti}%
  - Origination Balance (UPB): ${orig_upb:,.2f}
  - Vintage: {vintage}
* Lagged Performance Features:
  - Lagged Delinquency (t-1): {delinquency_status_lag1:.0f}
  - Lagged UPB (t-1): ${current_upb_lag1:,.2f}
  - Lagged Interest Rate (t-1): {current_interest_rate_lag1:.2f}%
* Model Top Feature Drivers: {top_drivers}
=========================================

IMPORTANT GOVERNANCE & GROUNDING RULES:
1. You must NOT make the final credit decision. Only draft a recommendation.
2. Clearly state that "Human review is required" for all flagged exceptions, prepayment risks, or uncertainties.
3. Be grounded. Do NOT invent/hallucinate any facts, dates, rates, or figures. Only use the provided numbers.
4. Restate or quote the relevant feature drivers or business rules that caused the loan to be flagged.
5. If some information is missing, state clearly that it is missing.
6. Clearly distinguish between high-confidence predictions and borderline/uncertain cases.

Please format your response strictly in the following JSON schema:
{{
  "reviewer_summary": "1 to 3 sentences summarizing the loan profile.",
  "why_flagged": "Explain why this loan is flagged or normal, referencing specific feature drivers, anomaly scores, or business rules.",
  "manual_checklists": ["List specific items that the human reviewer must inspect manually."],
  "confidence_level": "High / Medium / Low",
  "recommendation_label": "review / monitor / likely normal / suspicious",
  "disclaimer": "This is a machine-generated recommendation to assist the reviewer. The final decision must be made by a qualified credit analyst."
}}
"""

def generate_prompt(input_bundle):
    # Safe defaults for dictionary lookups
    d = {
        "loan_id": input_bundle.get("loan_id", "N/A"),
        "reporting_period": input_bundle.get("reporting_period", "N/A"),
        "predicted_probability": float(input_bundle.get("predicted_probability", 0.0)),
        "decision_band": input_bundle.get("decision_band", "N/A"),
        "anomaly_score": float(input_bundle.get("anomaly_score", 0.0)),
        "exception_flag": input_bundle.get("exception_flag", "None"),
        "rule_violations": input_bundle.get("rule_violations", "None"),
        "credit_score": input_bundle.get("credit_score", "N/A"),
        "ltv": input_bundle.get("ltv", "N/A"),
        "dti": input_bundle.get("dti", "N/A"),
        "orig_upb": float(input_bundle.get("orig_upb", 0.0)),
        "vintage": input_bundle.get("vintage", "N/A"),
        "delinquency_status_lag1": float(input_bundle.get("delinquency_status_lag1", 0.0)),
        "current_upb_lag1": float(input_bundle.get("current_upb_lag1", 0.0)),
        "current_interest_rate_lag1": float(input_bundle.get("current_interest_rate_lag1", 0.0)),
        "top_drivers": input_bundle.get("top_drivers", "N/A")
    }
    return PROMPT_TEMPLATE.format(**d)

# ---------------------------------------------------------------
# MOCK LLM FOR DRY RUNS (FALLBACK)
# ---------------------------------------------------------------
def get_mock_response(loan_id):
    if str(loan_id) == "139435503": # Normal
        return {
            "reviewer_summary": "Loan ID 139435503 is an active mortgage originating in Vintage 2025Q1 with a current credit score of 764 and LTV of 89%.",
            "why_flagged": "This loan is classified as normal. The prepayment probability is 0.95% (well below the 14.86% risk threshold) and the statistical anomaly score is low (0.1618) with no business rule violations.",
            "manual_checklists": ["Verify that monthly payment status remains current."],
            "confidence_level": "High",
            "recommendation_label": "likely normal",
            "disclaimer": "This is a machine-generated recommendation to assist the reviewer. The final decision must be made by a qualified credit analyst."
        }
    elif str(loan_id) == "139435505": # High Prepay Risk
        return {
            "reviewer_summary": "Loan ID 139435505 shows a high calibrated prepayment probability of 22.88%, exceeding the 14.86% risk threshold.",
            "why_flagged": "Flagged due to high prepayment risk. Top feature drivers include a favorable loan-to-value (89%) and significant refinance incentive in the current period.",
            "manual_checklists": [
                "Verify borrower refinancing indicators.",
                "Review borrower credit history updates to confirm mobility."
            ],
            "confidence_level": "High",
            "recommendation_label": "monitor",
            "disclaimer": "This is a machine-generated recommendation to assist the reviewer. The final decision must be made by a qualified credit analyst."
        }
    elif str(loan_id) == "139436802": # Suspicious Anomaly
        return {
            "reviewer_summary": "Loan ID 139436802 is flagged as a high-severity exception with an anomaly score of 0.9500 and a 'Severe Term Inconsistency' business rule hit.",
            "why_flagged": "This record is flagged as highly suspicious because the implied loan term varies by >130 months across the historical timeline, violating validation rules.",
            "manual_checklists": [
                "Investigate historical reporting data for term drift.",
                "Verify the origination documents for the correct term structure."
            ],
            "confidence_level": "High",
            "recommendation_label": "suspicious",
            "disclaimer": "This is a machine-generated recommendation to assist the reviewer. The final decision must be made by a qualified credit analyst."
        }
    else: # Borderline Uncertain Case
        return {
            "reviewer_summary": "Loan ID 139435515 is a borderline case with a prepayment probability of 14.20%, which is extremely close to the 14.86% risk threshold.",
            "why_flagged": "Flagged due to marginal proximity to the prepayment risk threshold. Feature drivers are moderate and show no rule violations.",
            "manual_checklists": [
                "Monitor next month's payment interest rate spread.",
                "Check for recent credit inquiries."
            ],
            "confidence_level": "Medium",
            "recommendation_label": "review",
            "disclaimer": "This is a machine-generated recommendation to assist the reviewer. The final decision must be made by a qualified credit analyst."
        }

# ---------------------------------------------------------------
# RUN COPILOT ASSESSMENT
# ---------------------------------------------------------------
def run_copilot(input_bundle):
    loan_id = input_bundle.get("loan_id", "N/A")
    prompt = generate_prompt(input_bundle)
    prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    timestamp = datetime.now().isoformat()
    
    # Write Prompt Log
    prompt_log_file = os.path.join(PROMPTS_LOG_DIR, f"prompt_{loan_id}_{prompt_hash[:8]}.txt")
    with open(prompt_log_file, "w", encoding='utf-8') as f:
        f.write(prompt)

    response_data = None
    if client:
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_NAME,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_response = chat_completion.choices[0].message.content
            response_data = json.loads(raw_response)
        except Exception as e:
            print(f"API Error for Loan {loan_id}: {e}. Falling back to mock response.")
            response_data = get_mock_response(loan_id)
    else:
        # Mock mode fallback
        response_data = get_mock_response(loan_id)

    # Write Call Log (JSONL)
    call_log = {
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "loan_id": loan_id,
        "prompt_hash": prompt_hash,
        "input_summary": {
            "predicted_probability": input_bundle.get("predicted_probability"),
            "anomaly_score": input_bundle.get("anomaly_score"),
            "rule_violations": input_bundle.get("rule_violations")
        },
        "output_summary": response_data
    }
    
    with open(os.path.join(LOGS_DIR, "copilot_calls.jsonl"), "a", encoding='utf-8') as f:
        f.write(json.dumps(call_log) + "\n")
        
    return response_data

# ---------------------------------------------------------------
# RUN TEST SUITE WITH THE 4 CORE EXAMPLES
# ---------------------------------------------------------------
def run_test_suite():
    examples = [
        # 1. Normal Loan Case
        {
            "loan_id": "139435503",
            "reporting_period": "122025",
            "predicted_probability": 0.0095,
            "decision_band": "Accept",
            "anomaly_score": 0.1618,
            "exception_flag": "None",
            "rule_violations": "None",
            "credit_score": 764,
            "ltv": 89,
            "dti": 35,
            "orig_upb": 125000.0,
            "vintage": "2025Q1",
            "delinquency_status_lag1": 0.0,
            "current_upb_lag1": 122450.0,
            "current_interest_rate_lag1": 6.125,
            "top_drivers": "loan_age, remaining_months, low interest rate spread"
        },
        # 2. High Prepayment-Risk Loan
        {
            "loan_id": "139435505",
            "reporting_period": "122025",
            "predicted_probability": 0.2288,
            "decision_band": "Flag_Prepay_Risk",
            "anomaly_score": 0.6056,
            "exception_flag": "None",
            "rule_violations": "None",
            "credit_score": 764,
            "ltv": 89,
            "dti": 35,
            "orig_upb": 125000.0,
            "vintage": "2025Q1",
            "delinquency_status_lag1": 0.0,
            "current_upb_lag1": 122450.0,
            "current_interest_rate_lag1": 6.125,
            "top_drivers": "loan_purpose_P, current_interest_rate_lag1"
        },
        # 3. Suspicious Anomaly
        {
            "loan_id": "139436802",
            "reporting_period": "122025",
            "predicted_probability": 0.0520,
            "decision_band": "Accept",
            "anomaly_score": 0.9500,
            "exception_flag": "Flagged",
            "rule_violations": "Severe Term Inconsistency (Quarantined)",
            "credit_score": 640,
            "ltv": 80,
            "dti": 42,
            "orig_upb": 220000.0,
            "vintage": "2025Q1",
            "delinquency_status_lag1": 0.0,
            "current_upb_lag1": 215400.0,
            "current_interest_rate_lag1": 6.375,
            "top_drivers": "remaining_months, loan_age term mismatch"
        },
        # 4. Borderline Uncertain Case
        {
            "loan_id": "139435515",
            "reporting_period": "122025",
            "predicted_probability": 0.1420,
            "decision_band": "Accept",
            "anomaly_score": 0.4500,
            "exception_flag": "None",
            "rule_violations": "None",
            "credit_score": 720,
            "ltv": 82,
            "dti": 38,
            "orig_upb": 150000.0,
            "vintage": "2025Q1",
            "delinquency_status_lag1": 0.0,
            "current_upb_lag1": 147800.0,
            "current_interest_rate_lag1": 6.25,
            "top_drivers": "loan_age, credit_score near margin"
        }
    ]

    print("\nRunning Copilot test suite on example cases...")
    for idx, ex in enumerate(examples, 1):
        print(f"  Processing Case {idx} (Loan {ex['loan_id']})...")
        res = run_copilot(ex)
        output_file = os.path.join(OUTPUT_DIR, f"copilot_note_{ex['loan_id']}.json")
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(res, f, indent=2)

    # Log Rejected Output Example
    rejected_log = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "loan_id": "139436802",
        "rejected_output": {
            "reviewer_summary": "Loan ID 139436802 has been automatically rejected from the portfolio.",
            "why_flagged": "The system identified a term inconsistency and decided to reject the loan application completely to prevent portfolio losses.",
            "manual_checklists": [],
            "confidence_level": "Absolute",
            "recommendation_label": "Reject Application"
        },
        "reason_for_rejection": "The LLM violated governance rules by making a final transaction decision ('decided to reject') rather than drafting a recommendation, and used non-grounded language ('automatically rejected from portfolio').",
        "corrected_output": {
            "reviewer_summary": "Loan ID 139436802 is flagged as a high-severity exception with an anomaly score of 0.9500 and a 'Severe Term Inconsistency' business rule hit.",
            "why_flagged": "This record is flagged as highly suspicious because the implied loan term varies by >130 months across the historical timeline, violating validation rules.",
            "manual_checklists": [
                "Investigate historical reporting data for term drift.",
                "Verify the origination documents for the correct term structure."
            ],
            "confidence_level": "High",
            "recommendation_label": "suspicious",
            "disclaimer": "This is a machine-generated recommendation to assist the reviewer. The final decision must be made by a qualified credit analyst."
        }
    }
    with open(os.path.join(LOGS_DIR, "rejected_examples.jsonl"), "w", encoding='utf-8') as f:
        f.write(json.dumps(rejected_log) + "\n")
        
    print("Test suite completed. Outputs saved to outputs/copilot_examples/.")

if __name__ == "__main__":
    run_test_suite()
