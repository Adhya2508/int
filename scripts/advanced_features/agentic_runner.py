"""
=================================================================
INTAIN AI CHALLENGE -- ADVANCED FEATURES: AGENTIC RUNNER
Loan Performance Intelligence Engine
Intain Campus FinTech Challenge 2026

This script acts as the central orchestrator (Agentic Runner). It:
  - Executes profiling, modeling, survival, scenario, and copilot runs.
  - Collects run logs, execution statuses, and generated outputs.
  - Keeps track of run history inside logs/experiment_tracking/
=================================================================
"""
import os
import sys
import subprocess
import json
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = "e:/intain"
RUNNER_LOG = os.path.join(BASE_DIR, "logs/experiment_tracking/runner_runs.jsonl")
os.makedirs(os.path.dirname(RUNNER_LOG), exist_ok=True)

def run_step(step_name, command):
    print(f"\n==========================================")
    print(f"RUNNING STEP: {step_name}")
    print(f"==========================================")
    timestamp_start = datetime.now().isoformat()
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(res.stdout)
        status = "SUCCESS"
        error_msg = ""
    except subprocess.CalledProcessError as e:
        print(f"Error in {step_name}: {e.stderr}")
        status = "FAILED"
        error_msg = e.stderr
    
    timestamp_end = datetime.now().isoformat()
    
    log_entry = {
        "step_name": step_name,
        "command": command,
        "status": status,
        "started_at": timestamp_start,
        "ended_at": timestamp_end,
        "error": error_msg
    }
    
    with open(RUNNER_LOG, "a", encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return status

def main():
    steps = [
        ("Task 1-6 Master Pipeline", "py e:/intain/scripts/run_all_tasks.py"),
        ("Task 7 Reviewer Copilot Test", "py e:/intain/scripts/llm_copilot/copilot.py"),
        ("Advanced Statistical Pipelines", "py e:/intain/scripts/advanced_features/run_advanced_features.py"),
        ("RAG Grounded Search Test", "py e:/intain/scripts/advanced_features/rag_assistant.py 'credit_score'")
    ]
    
    print("Agentic Experiment Runner initiated...")
    all_status = {}
    for name, cmd in steps:
        status = run_step(name, cmd)
        all_status[name] = status
        
    print("\n==========================================")
    print("RUNNER SUMMARY:")
    print("==========================================")
    for name, status in all_status.items():
        print(f"- {name}: {status}")

if __name__ == "__main__":
    main()
