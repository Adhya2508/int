"""
=================================================================
INTAIN AI CHALLENGE -- ADVANCED FEATURES: DASHBOARD GENERATOR
Loan Performance Intelligence Engine
Intain Campus FinTech Challenge 2026

This script reads all JSON calculations and compiles them into a
beautiful, self-contained, interactive HTML dashboard.
=================================================================
"""
import os
import sys
import json
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = "e:/intain"
OUT_DIR  = os.path.join(BASE_DIR, "dashboard/advanced_features")
os.makedirs(OUT_DIR, exist_ok=True)

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def main():
    print("Generating HTML Dashboard...")
    
    # Load all advanced metric results
    comp_risk    = load_json("e:/intain/outputs/advanced_features/competing_risk.json")
    monte_carlo  = load_json("e:/intain/outputs/monte_carlo/portfolio_simulation.json")
    drift        = load_json("e:/intain/outputs/monitoring/drift_metrics.json")
    segment      = load_json("e:/intain/outputs/advanced_features/segment_scenario_curves.json")
    calibration  = load_json("e:/intain/outputs/calibration/calibration_by_segment.json")
    fairness     = load_json("e:/intain/outputs/fairness/fairness_metrics.json")
    counterfact  = load_json("e:/intain/outputs/counterfactuals/counterfactual_examples.json")
    sensitivity  = load_json("e:/intain/outputs/advanced_features/stress_sensitivity.json")
    ci           = load_json("e:/intain/outputs/confidence_intervals/confidence_intervals.json")
    synthetic    = load_json("e:/intain/outputs/advanced_features/synthetic_stress_test.json")
    
    # Load RAG documents
    rag_docs = []
    dp_path = "e:/intain/data_dictionary.md"
    if os.path.exists(dp_path):
        with open(dp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("*") and ":" in line:
                    rag_docs.append({"source": "data_dictionary.md", "content": line})
                    
    vr_path = "e:/intain/validation_rules.json"
    if os.path.exists(vr_path):
        with open(vr_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
            for k, v in rules.items():
                rag_docs.append({"source": "validation_rules.json", "content": f"{k}: {json.dumps(v)}"})

    # Load Active learning review queue
    al_queue = []
    al_path = "e:/intain/outputs/advanced_features/active_learning_queue.csv"
    if os.path.exists(al_path):
        df_al = pd.read_csv(al_path).head(10)
        al_queue = df_al.to_dict(orient="records")

    # Embed data as JS objects
    data_js = f"""
    const COMP_RISK = {json.dumps(comp_risk)};
    const MONTE_CARLO = {json.dumps(monte_carlo)};
    const DRIFT = {json.dumps(drift)};
    const SEGMENT = {json.dumps(segment)};
    const CALIBRATION = {json.dumps(calibration)};
    const FAIRNESS = {json.dumps(fairness)};
    const COUNTERFACT = {json.dumps(counterfact)};
    const SENSITIVITY = {json.dumps(sensitivity)};
    const CONF_INTERVAL = {json.dumps(ci)};
    const SYNTHETIC = {json.dumps(synthetic)};
    const RAG_DOCS = {json.dumps(rag_docs)};
    const AL_QUEUE = {json.dumps(al_queue)};
    """

    # HTML TEMPLATE
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Quality & Advanced Analytics Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #1e293b;
            --primary-light: #334155;
            --accent: #2563eb;
            --accent-hover: #1d4ed8;
            --bg-light: #f8fafc;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --card-bg: #ffffff;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
        }}
        
        body {{
            background-color: var(--bg-light);
            color: var(--text-dark);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }}
        
        /* HEADER (Speridian & Pragma style integration) */
        header {{
            background-color: var(--primary);
            color: white;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 3px solid var(--accent);
            box-shadow: var(--shadow);
        }}
        
        .logo-container {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .logo-icon {{
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            width: 2rem;
            height: 2rem;
            border-radius: 6px;
        }}
        
        .logo-text {{
            font-weight: 700;
            font-size: 1.25rem;
            letter-spacing: -0.5px;
        }}
        
        .nav-links {{
            display: flex;
            gap: 1.5rem;
            list-style: none;
        }}
        
        .nav-links a {{
            color: #cbd5e1;
            text-decoration: none;
            font-size: 0.95rem;
            font-weight: 500;
            transition: color 0.2s;
        }}
        
        .nav-links a:hover {{
            color: white;
        }}
        
        /* MAIN BODY LAYOUT */
        .dashboard-container {{
            display: flex;
            flex: 1;
        }}
        
        /* SIDEBAR */
        sidebar {{
            background-color: white;
            width: 260px;
            border-right: 1px solid var(--border);
            padding: 1.5rem 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        
        .sidebar-btn {{
            background: none;
            border: none;
            text-align: left;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .sidebar-btn:hover {{
            background-color: var(--bg-light);
            color: var(--accent);
        }}
        
        .sidebar-btn.active {{
            background-color: #eff6ff;
            color: var(--accent);
        }}
        
        /* CONTENT SECTION */
        main {{
            flex: 1;
            padding: 2rem;
            overflow-y: auto;
            max-width: 1400px;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }}
        
        /* CARDS & GRID */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
        }}
        
        .metric-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            transition: transform 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-dark);
        }}
        
        .metric-trend {{
            font-size: 0.85rem;
            font-weight: 600;
        }}
        
        .trend-up {{ color: #10b981; }}
        .trend-down {{ color: #ef4444; }}
        
        /* ANALYTICS SECTION GRID */
        .chart-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}
        
        @media (max-width: 900px) {{
            .chart-row {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .chart-card {{
            background-color: white;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        
        .chart-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-dark);
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
        }}
        
        /* RAG STYLING */
        .search-container {{
            display: flex;
            gap: 1rem;
            width: 100%;
        }}
        
        .search-input {{
            flex: 1;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            font-size: 1rem;
            outline: none;
        }}
        
        .search-input:focus {{
            border-color: var(--accent);
        }}
        
        .search-btn {{
            background-color: var(--accent);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        
        .search-btn:hover {{
            background-color: var(--accent-hover);
        }}
        
        .results-list {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .result-item {{
            background-color: white;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: var(--shadow);
        }}
        
        .result-source {{
            font-size: 0.75rem;
            color: var(--accent);
            font-weight: 700;
            text-transform: uppercase;
        }}
        
        .result-text {{
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }}
        
        /* ACTIVE LEARNING TABLE */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.5rem;
        }}
        
        th, td {{
            text-align: left;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            background-color: var(--bg-light);
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
        }}
        
        td {{
            font-size: 0.9rem;
        }}
        
        tr:hover {{
            background-color: var(--bg-light);
        }}
        
        .badge {{
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        
        .badge-high {{ background-color: #fee2e2; color: #ef4444; }}
        .badge-medium {{ background-color: #fef3c7; color: #d97706; }}
        .badge-low {{ background-color: #d1fae5; color: #10b981; }}
    </style>
</head>
<body>

    <!-- HEADER -->
    <header>
        <div class="logo-container">
            <div class="logo-icon"></div>
            <span class="logo-text">pragma loan analytics</span>
        </div>
        <ul class="nav-links">
            <li><a href="#summary">Dashboard</a></li>
            <li><a href="#dictionary">Dictionary</a></li>
            <li><a href="#about">About System</a></li>
        </ul>
    </header>

    <div class="dashboard-container">
        <!-- SIDEBAR -->
        <sidebar>
            <button class="sidebar-btn active" onclick="switchTab('summary', this)">Summary Metrics</button>
            <button class="sidebar-btn" onclick="switchTab('competing', this)">Competing Risk Model</button>
            <button class="sidebar-btn" onclick="switchTab('drift', this)">Drift Monitoring</button>
            <button class="sidebar-btn" onclick="switchTab('fairness', this)">Fairness & Bias</button>
            <button class="sidebar-btn" onclick="switchTab('active-learning', this)">Active Learning Queue</button>
            <button class="sidebar-btn" onclick="switchTab('rag-search', this)">Grounded RAG Search</button>
        </sidebar>

        <!-- MAIN CONTENT PANEL -->
        <main>
            <!-- TAB: SUMMARY METRICS -->
            <div id="summary" class="tab-content active">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <span class="metric-label">Monte Carlo Median Prepayment</span>
                        <span class="metric-value" id="val-mc-median">--</span>
                        <span class="metric-trend trend-up">Calibrated</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-label">Validation ROC-AUC</span>
                        <span class="metric-value">0.8116</span>
                        <span class="metric-trend trend-up">High Accuracy</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-label">95% Confidence Interval (AUC)</span>
                        <span class="metric-value" id="val-auc-ci">--</span>
                        <span class="metric-trend trend-up">Bootstrapped</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-label">Synthetic Stress Prepay</span>
                        <span class="metric-value" id="val-synthetic">--</span>
                        <span class="metric-trend trend-down">Extreme Stress</span>
                    </div>
                </div>

                <div class="chart-row">
                    <div class="chart-card">
                        <span class="chart-title">Stress Sensitivity by Feature Cluster</span>
                        <canvas id="chartSensitivity" style="max-height: 250px;"></canvas>
                    </div>
                    <div class="chart-card">
                        <span class="chart-title">Calibration Error (Brier Score) per Credit Band</span>
                        <canvas id="chartCalibration" style="max-height: 250px;"></canvas>
                    </div>
                </div>
            </div>

            <!-- TAB: COMPETING RISK -->
            <div id="competing" class="tab-content">
                <div class="chart-card" style="width: 100%;">
                    <span class="chart-title">Competing Risk Cumulative Incidence Functions (CIF)</span>
                    <canvas id="chartCompeting" style="max-height: 400px;"></canvas>
                </div>
            </div>

            <!-- TAB: DRIFT MONITORING -->
            <div id="drift" class="tab-content">
                <div class="chart-card" style="width: 100%;">
                    <span class="chart-title">Population Stability Index (PSI) -- Train vs Test Drift</span>
                    <canvas id="chartDrift" style="max-height: 400px;"></canvas>
                </div>
            </div>

            <!-- TAB: FAIRNESS -->
            <div id="fairness" class="tab-content">
                <div class="chart-card">
                    <span class="chart-title">Fairness Metrics (True Positive Rate by Credit Band)</span>
                    <table>
                        <thead>
                            <tr>
                                <th>Segment (Credit Band)</th>
                                <th>True Positive Rate (Equal Opportunity)</th>
                                <th>Brier Score Calibration</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Prime (Credit Score > 720)</td>
                                <td id="fair-prime-tpr">--</td>
                                <td id="fair-prime-brier">--</td>
                            </tr>
                            <tr>
                                <td>Near-Prime (Credit Score 660-720)</td>
                                <td id="fair-near-tpr">--</td>
                                <td id="fair-near-brier">--</td>
                            </tr>
                            <tr>
                                <td>Subprime (Credit Score < 660)</td>
                                <td id="fair-subprime-tpr">--</td>
                                <td id="fair-subprime-brier">--</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- TAB: ACTIVE LEARNING -->
            <div id="active-learning" class="tab-content">
                <div class="chart-card">
                    <span class="chart-title">Active Learning Priority Review Queue (Top 10 High-Priority Loans)</span>
                    <table>
                        <thead>
                            <tr>
                                <th>Loan ID</th>
                                <th>Credit Score</th>
                                <th>LTV</th>
                                <th>DTI</th>
                                <th>Base Probability</th>
                                <th>Anomaly Score</th>
                                <th>Priority</th>
                            </tr>
                        </thead>
                        <tbody id="al-table-body">
                            <!-- Injected dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- TAB: RAG SEARCH -->
            <div id="rag-search" class="tab-content">
                <div class="chart-card">
                    <span class="chart-title">Grounded RAG Assistant (Search Dictionary & Validation Rules)</span>
                    <div class="search-container">
                        <input type="text" id="ragQuery" class="search-input" placeholder="Type a keyword (e.g. ltv, delinquency, balance_consistency)...">
                        <button class="search-btn" onclick="executeRAG()">Search</button>
                    </div>
                    <div id="ragResults" class="results-list">
                        <!-- Injected dynamically -->
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- EMBEDDED DATA -->
    <script>
        {data_js}
        
        // SWITCH TAB FUNCTION
        function switchTab(tabId, btn) {{
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.remove('active'));
            
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');
        }}

        // LOAD METRIC LABELS
        document.getElementById('val-mc-median').innerText = (MONTE_CARLO.median_prepayment_rate * 100).toFixed(2) + "%";
        document.getElementById('val-auc-ci').innerText = "[" + CONF_INTERVAL.roc_auc_ci[0].toFixed(4) + ", " + CONF_INTERVAL.roc_auc_ci[1].toFixed(4) + "]";
        document.getElementById('val-synthetic').innerText = (SYNTHETIC.synthetic_prepay_probability * 100).toFixed(2) + "%";

        // LOAD FAIRNESS TABLES
        document.getElementById('fair-prime-tpr').innerText = (FAIRNESS.true_positive_rates.Prime * 100).toFixed(2) + "%";
        document.getElementById('fair-near-tpr').innerText = (FAIRNESS.true_positive_rates['Near-Prime'] * 100).toFixed(2) + "%";
        document.getElementById('fair-subprime-tpr').innerText = (FAIRNESS.true_positive_rates.Subprime * 100).toFixed(2) + "%";
        
        document.getElementById('fair-prime-brier').innerText = CALIBRATION.Prime.brier_score.toFixed(4);
        document.getElementById('fair-near-brier').innerText = CALIBRATION['Near-Prime'].brier_score.toFixed(4);
        document.getElementById('fair-subprime-brier').innerText = CALIBRATION.Subprime.brier_score.toFixed(4);

        // LOAD ACTIVE LEARNING QUEUE TABLE
        const tbody = document.getElementById('al-table-body');
        AL_QUEUE.forEach(row => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${{row.loan_id}}</td>
                <td>${{row.credit_score}}</td>
                <td>${{row.ltv}}%</td>
                <td>${{row.dti}}%</td>
                <td>${{(row.prob_base * 100).toFixed(2)}}%</td>
                <td>${{row.anomaly_score.toFixed(4)}}</td>
                <td><span class="badge badge-high">${{row.active_learning_priority.toFixed(4)}}</span></td>
            `;
            tbody.appendChild(tr);
        }});

        // GROUNDED RAG SEARCH
        function executeRAG() {{
            const query = document.getElementById('ragQuery').value.toLowerCase();
            const resultsBox = document.getElementById('ragResults');
            resultsBox.innerHTML = '';
            
            if (!query.trim()) return;
            
            let matched = [];
            RAG_DOCS.forEach(doc => {{
                let score = 0;
                query.split(' ').forEach(word => {{
                    if (doc.content.toLowerCase().includes(word)) score++;
                }});
                if (score > 0) {{
                    matched.push({{ score, doc }});
                }}
            }});
            
            matched.sort((a, b) => b.score - a.score);
            const topHits = matched.slice(0, 3);
            
            if (topHits.length === 0) {{
                resultsBox.innerHTML = '<div class="result-item">No grounded definitions found.</div>';
                return;
            }}
            
            topHits.forEach(hit => {{
                const item = document.createElement('div');
                item.className = 'result-item';
                item.innerHTML = `
                    <div class="result-source">${{hit.doc.source}}</div>
                    <div class="result-text">${{hit.doc.content}}</div>
                `;
                resultsBox.appendChild(item);
            }});
        }}

        // CHART: STRESS SENSITIVITY
        const ctxSensitivity = document.getElementById('chartSensitivity').getContext('2d');
        new Chart(ctxSensitivity, {{
            type: 'bar',
            data: {{
                labels: ['Base Prepayment Rate', 'Credit Stress (-50 FICO)', 'Interest Rate Stress (+2.0%)'],
                datasets: [{{
                    label: 'Mean Probability',
                    data: [SENSITIVITY.base_probability, SENSITIVITY.credit_stress_probability, SENSITIVITY.rate_stress_probability],
                    backgroundColor: ['#64748b', '#ef4444', '#3b82f6'],
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});

        // CHART: CALIBRATION
        const ctxCalibration = document.getElementById('chartCalibration').getContext('2d');
        new Chart(ctxCalibration, {{
            type: 'bar',
            data: {{
                labels: ['Subprime', 'Near-Prime', 'Prime'],
                datasets: [{{
                    label: 'Brier Loss',
                    data: [CALIBRATION.Subprime.brier_score, CALIBRATION['Near-Prime'].brier_score, CALIBRATION.Prime.brier_score],
                    backgroundColor: '#10b981',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});

        // CHART: COMPETING RISK
        const ctxCompeting = document.getElementById('chartCompeting').getContext('2d');
        new Chart(ctxCompeting, {{
            type: 'line',
            data: {{
                labels: COMP_RISK.months.map(m => "M" + m),
                datasets: [
                    {{
                        label: 'Prepayment Cumulative Incidence (CIF)',
                        data: COMP_RISK.CIF_prepayment,
                        borderColor: '#2563eb',
                        fill: false
                    }},
                    {{
                        label: 'Maturity Cumulative Incidence (CIF)',
                        data: COMP_RISK.CIF_maturity,
                        borderColor: '#10b981',
                        fill: false
                    }},
                    {{
                        label: 'Survival Probability S(t)',
                        data: COMP_RISK.survival_prob,
                        borderColor: '#64748b',
                        fill: false
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});

        // CHART: DRIFT
        const ctxDrift = document.getElementById('chartDrift').getContext('2d');
        new Chart(ctxDrift, {{
            type: 'bar',
            data: {{
                labels: Object.keys(DRIFT),
                datasets: [{{
                    label: 'PSI Value',
                    data: Object.values(DRIFT),
                    backgroundColor: '#3b82f6',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});
    </script>
</body>
</html>
"""

    # Write dashboard file
    output_html = os.path.join(OUT_DIR, "dashboard.html")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Dashboard generated successfully at {output_html}!")

if __name__ == "__main__":
    main()
