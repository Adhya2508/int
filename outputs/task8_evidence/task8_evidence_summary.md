# Task 8 Evidence Summary Log

This log lists the exact repository file paths where the agentic ML development evidence for the **Intain Campus FinTech Challenge 2026** is stored:

1. **Development Log & Report:**
   - [`reports/task8_ai_development/ai_development_log.md`](file:///e:/intain/reports/task8_ai_development/ai_development_log.md) (The main registry detailing stages, rejections, code shares, and reviews).
2. **Rejections & Corrections Log:**
   - [`logs/rejections/rejection_log.jsonl`](file:///e:/intain/logs/rejections/rejection_log.jsonl) (Structured JSONL log of rejected prompts, feature leakage, and metric calibrations).
3. **LLM Copilot Rejections Log:**
   - [`logs/copilot/rejected_examples.jsonl`](file:///e:/intain/logs/copilot/rejected_examples.jsonl) (Documents the specific case where LLM overconfidence or decision-making was corrected to a recommendation label).
4. **Prompt Audit Trail:**
   - [`logs/prompts/`](file:///e:/intain/logs/prompts) (Contains raw text prompt logs generated during LLM Copilot test runs).
5. **Execution Logs:**
   - [`logs/copilot/copilot_calls.jsonl`](file:///e:/intain/logs/copilot/copilot_calls.jsonl) (Historical JSONL database tracking every copilot prompt hash, inputs, and outputs).
6. **Task Execution Code:**
   - [`scripts/run_all_tasks.py`](file:///e:/intain/scripts/run_all_tasks.py) (The complete reproducible script executing Tasks 1, 3, 4, 5, and 6).
   - [`scripts/llm_copilot/copilot.py`](file:///e:/intain/scripts/llm_copilot/copilot.py) (The complete reviewer copilot with fallback mock templates and dotenv loaders).
