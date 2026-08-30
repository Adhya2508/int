"""
=================================================================
INTAIN AI CHALLENGE -- ADVANCED FEATURES: RAG ASSISTANT
Loan Performance Intelligence Engine
Intain Campus FinTech Challenge 2026

This script implements a local, fully-grounded search engine
(RAG proxy) over:
  - e:/intain/data_dictionary.md
  - e:/intain/validation_rules.json
=================================================================
"""
import os
import sys
import json
import re
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = "e:/intain"
RAG_LOG_FILE = os.path.join(BASE_DIR, "logs/rag/rag_log.jsonl")
os.makedirs(os.path.dirname(RAG_LOG_FILE), exist_ok=True)

class LocalRAG:
    def __init__(self):
        self.documents = []
        self.load_data_dictionary()
        self.load_validation_rules()

    def load_data_dictionary(self):
        path = os.path.join(BASE_DIR, "data_dictionary.md")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
                for idx, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith("*") and ":" in line:
                        self.documents.append({
                            "source": "data_dictionary.md",
                            "line_num": idx + 1,
                            "content": f"Data Dictionary Row: {line}"
                        })

    def load_validation_rules(self):
        path = os.path.join(BASE_DIR, "validation_rules.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    rules = json.load(f)
                    if isinstance(rules, dict):
                        # Flatten dictionary to list of rules
                        for key, val in rules.items():
                            self.documents.append({
                                "source": "validation_rules.json",
                                "key": key,
                                "content": f"Validation Rule [{key}]: {json.dumps(val)}"
                            })
            except Exception as e:
                print(f"Error loading validation rules JSON: {e}")

    def query(self, search_text):
        search_text = search_text.lower()
        results = []
        for doc in self.documents:
            # Score based on keyword overlap
            words = search_text.split()
            score = 0
            for w in words:
                if w in doc["content"].lower():
                    score += 1
            if score > 0:
                results.append((score, doc))
        
        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        top_hits = results[:3]
        
        # Log query
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": search_text,
            "top_hits": [hit[1] for hit in top_hits]
        }
        with open(RAG_LOG_FILE, "a", encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")
            
        return top_hits

if __name__ == "__main__":
    assistant = LocalRAG()
    query_str = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "credit_score"
    print(f"\nQuerying grounded database for: '{query_str}'...")
    hits = assistant.query(query_str)
    if not hits:
        print("No matching definitions or rules found.")
    for idx, (score, doc) in enumerate(hits, 1):
        print(f"\nHit {idx} (Score: {score}) from {doc['source']}:")
        print(doc["content"])
