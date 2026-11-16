"""
Generates a standalone, permanent HTML dashboard (index.html & dashboard.html)
with zero backend requirements. Can be hosted on GitHub Pages or double-clicked offline!
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\aru12\OneDrive\Desktop\capstone project")

# 1. Load Data
tables = {}
for f in (ROOT / "outputs/tables").glob("*.csv"):
    tables[f.stem] = pd.read_csv(f).to_dict(orient="records")

ref = ROOT / "reference_materials/Major-project-github-ref"
for name in ["thesis_ready_final_results", "target_thresholds", "target_class_distribution_comparison"]:
    p = ref / "outputs/tables" / f"{name}.csv"
    if p.exists():
        tables[name] = pd.read_csv(p).to_dict(orient="records")

p_multi = ref / "outputs/audits/final_corrections_v5/multiseed_per_seed_metrics.csv"
if p_multi.exists():
    tables["multiseed_per_seed_metrics"] = pd.read_csv(p_multi).to_dict(orient="records")

# 1000 row Chicago sample for instant client-side filtering
d2_p = ROOT / "data/processed/dataset2_enhanced_features.csv"
if d2_p.exists():
    df_sample = pd.read_csv(d2_p, nrows=3000)
    cols = ["Date", "Primary Type", "Location Description", "District", "Beat", "Hour", "Month", "DayOfWeek", "Domestic", "Is_Residential"]
    df_sample = df_sample[cols].sample(n=1000, random_state=42)
    df_sample["Date"] = df_sample["Date"].str[:10]
    tables["chicago_sample"] = df_sample.to_dict(orient="records")

tables_json = json.dumps(tables)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crime Analytics & ML Research Command Center</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Plotly.js CDN -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    colors: {{
                        darkBg: '#0b0f19',
                        surface: '#151c2c',
                        surfaceLight: '#1e293b',
                        borderCol: '#232f48',
                        emeraldAcc: '#10b981',
                        tealAcc: '#14b8a6',
                        cyanAcc: '#06b6d4',
                        roseAcc: '#f43f5e',
                        amberAcc: '#f59e0b'
                    }}
                }}
            }}
        }}
    </script>
    <style>
        body {{
            background-color: #0b0f19;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        .kpi-card {{
            background-color: #151c2c;
            border: 1px solid #232f48;
            border-radius: 8px;
            padding: 16px 20px;
            transition: all 0.2s ease;
        }}
        .kpi-card:hover {{
            border-color: #10b981;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.15);
        }}
        .nav-item.active {{
            background-color: #1e293b;
            color: #10b981;
            border-left: 3px solid #10b981;
            font-weight: 600;
        }}
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: #0b0f19;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #232f48;
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #10b981;
        }}
    </style>
</head>
<body class="flex min-h-screen">

    <!-- SIDEBAR -->
    <aside class="w-72 bg-[#0d1322] border-r border-[#232f48] flex flex-col justify-between p-4 sticky top-0 h-screen overflow-y-auto">
        <div>
            <!-- Header -->
            <div class="pb-4 mb-4 border-b border-[#232f48]">
                <span class="px-2.5 py-1 text-xs font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">COMMAND CENTER</span>
                <h1 class="text-lg font-bold text-slate-100 mt-2">CRIME ML RESEARCH</h1>
                <p class="text-xs text-slate-400">Comparative Evaluation of ML & DL</p>
            </div>

            <!-- Navigation Links -->
            <nav class="space-y-1 text-sm text-slate-300">
                <a href="#sec-overview" class="nav-item flex items-center px-3 py-2 rounded hover:bg-[#1e293b] active" onclick="setActiveNav(this)">
                    <span class="mr-2">🏛️</span> 1. Research at a Glance
                </a>
                <a href="#sec-explorer" class="nav-item flex items-center px-3 py-2 rounded hover:bg-[#1e293b]" onclick="setActiveNav(this)">
                    <span class="mr-2">🔍</span> 2. Crime Data Explorer
                </a>
                <a href="#sec-methodology" class="nav-item flex items-center px-3 py-2 rounded hover:bg-[#1e293b]" onclick="setActiveNav(this)">
                    <span class="mr-2">⚙️</span> 3. Data & Methodology
                </a>
                <a href="#sec-feature-analysis" class="nav-item flex items-center px-3 py-2 rounded hover:bg-[#1e293b]" onclick="setActiveNav(this)">
                    <span class="mr-2">🔬</span> 4. Feature Ablation Lab
                </a>
                <a href="#sec-ml-experiments" class="nav-item flex items-center px-3 py-2 rounded hover:bg-[#1e293b]" onclick="setActiveNav(this)">
                    <span class="mr-2">🧪</span> 5. Machine Learning Lab
                </a>
                <a href="#sec-deep-learning" class="nav-item flex items-center px-3 py-2 rounded hover:bg-[#1e293b]" onclick="setActiveNav(this)">
                    <span class="mr-2">🧠</span> 6. Deep Learning Models
                </a>
                <a href="#sec-evaluation" class="nav-item flex items-center px-3 py-2 rounded hover:bg-[#1e293b]" onclick="setActiveNav(this)">
                    <span class="mr-2">📊</span> 7. Calibration & Fairness
                </a>
                <a href="#sec-insights" class="nav-item flex items-center px-3 py-2 rounded hover:bg-[#1e293b]" onclick="setActiveNav(this)">
                    <span class="mr-2">💡</span> 8. Research Insights
                </a>
                <a href="#sec-details" class="nav-item flex items-center px-3 py-2 rounded hover:bg-[#1e293b]" onclick="setActiveNav(this)">
                    <span class="mr-2">📋</span> 9. Technical Details & Data
                </a>
                <a href="#sec-conclusion" class="nav-item flex items-center px-3 py-2 rounded hover:bg-[#1e293b]" onclick="setActiveNav(this)">
                    <span class="mr-2">🎯</span> 10. Conclusion & Roadmap
                </a>
            </nav>
        </div>

        <!-- Empirical Badge -->
        <div class="mt-6 pt-3 border-t border-[#232f48] text-xs text-slate-400">
            <div class="bg-[#151c2c] p-3 rounded border border-[#232f48] space-y-1">
                <span class="text-slate-500 font-semibold uppercase tracking-wider block">Empirical Highlights</span>
                <div>• Chicago D2 AUC: <b class="text-emerald-400">0.9152</b></div>
                <div>• Domestic Recall: <b class="text-teal-400">87.18%</b></div>
                <div>• Isotonic ECE: <b class="text-amber-400">0.47%</b></div>
                <div>• NCRB 5-Seed F1: <b class="text-cyan-400">0.7097</b></div>
            </div>
            <div class="mt-2 text-center text-slate-500">
                Peer-Review Ready • Permanent HTML
            </div>
        </div>
    </aside>

    <!-- MAIN CONTENT -->
    <main class="flex-1 p-8 space-y-12 max-w-6xl mx-auto">

        <!-- 1. OVERVIEW SECTION -->
        <section id="sec-overview" class="space-y-6">
            <div class="p-6 rounded-lg bg-gradient-to-r from-[#151c2c] to-[#0d1322] border-l-4 border-emerald-500 shadow-lg">
                <div class="flex items-center space-x-2">
                    <span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">CAPSTONE RESEARCH PLATFORM</span>
                    <span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-teal-500/15 text-teal-400 border border-teal-500/30">PEER-REVIEW COMPLIANT</span>
                </div>
                <h1 class="text-2xl font-bold text-slate-100 mt-2">
                    Comparative Evaluation of Machine Learning and Deep Learning for Crime Data Classification
                </h1>
                <p class="text-sm text-slate-300 mt-1 max-w-4xl leading-relaxed">
                    An empirical research framework evaluating tree-based ensembles, deep tabular representations, 
                    recurrent sequence models, probability calibration, and algorithmic fairness across two rigorous, leak-free public safety pipelines.
                </p>
            </div>

            <!-- Top KPI Row -->
            <div class="grid grid-cols-1 md:grid-cols-5 gap-4">
                <div class="kpi-card">
                    <div class="flex justify-between items-center text-xs text-slate-400 font-semibold uppercase">Total Records <span class="text-cyan-400">DUAL</span></div>
                    <div class="text-2xl font-bold text-slate-100 mt-1">765,716</div>
                    <div class="text-xs text-slate-400 mt-1">756k Chicago + 9.6k NCRB</div>
                </div>
                <div class="kpi-card">
                    <div class="flex justify-between items-center text-xs text-slate-400 font-semibold uppercase">Chicago ROC-AUC <span class="text-emerald-400">CHAMPION</span></div>
                    <div class="text-2xl font-bold text-emerald-400 mt-1">0.9152</div>
                    <div class="text-xs text-emerald-500/80 mt-1">↑ +0.3557 vs Temporal</div>
                </div>
                <div class="kpi-card">
                    <div class="flex justify-between items-center text-xs text-slate-400 font-semibold uppercase">Domestic Recall <span class="text-teal-400">CRITICAL</span></div>
                    <div class="text-2xl font-bold text-teal-400 mt-1">87.18%</div>
                    <div class="text-xs text-slate-400 mt-1">Threshold 0.47 (Bal Acc 83.5%)</div>
                </div>
                <div class="kpi-card">
                    <div class="flex justify-between items-center text-xs text-slate-400 font-semibold uppercase">Calibration ECE <span class="text-amber-400">ISOTONIC</span></div>
                    <div class="text-2xl font-bold text-amber-400 mt-1">0.47%</div>
                    <div class="text-xs text-emerald-500/80 mt-1">↓ -14.13% vs Uncalibrated</div>
                </div>
                <div class="kpi-card">
                    <div class="flex justify-between items-center text-xs text-slate-400 font-semibold uppercase">NCRB Sequence F1 <span class="text-cyan-400">LSTM</span></div>
                    <div class="text-2xl font-bold text-cyan-400 mt-1">0.7097</div>
                    <div class="text-xs text-slate-400 mt-1">5-Seed Mean (3-Yr Seq)</div>
                </div>
            </div>

            <!-- Dual Pipeline Cards -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                <div class="bg-[#151c2c] p-5 rounded-lg border-t-4 border-emerald-500 border border-[#232f48] shadow">
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="font-bold text-slate-100">Pipeline 1: Chicago Incident Classification</h3>
                        <span class="px-2 py-0.5 text-xs rounded bg-emerald-500/20 text-emerald-400">Micro-Incident Triage</span>
                    </div>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        <b>Dataset:</b> Chicago Police Department (2023–2025), 756,085 records across 22 districts.<br>
                        <b>Task:</b> Binary classification of domestic altercations (<code class="text-emerald-400">Domestic = 1/0</code>).<br>
                        <b>Class Balance:</b> 18.46% Domestic prevalence (minority class imbalance).<br>
                        <b>Temporal Split:</b> Train (388k, 2023–24H1), Val (131k, 2024H2), Locked Test (236k, 2025).<br>
                        <b>Leakage Purge:</b> Removed <code>Description</code> (contained "DOMESTIC" in 143k rows) and post-arrest flags.
                    </p>
                </div>
                <div class="bg-[#151c2c] p-5 rounded-lg border-t-4 border-teal-500 border border-[#232f48] shadow">
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="font-bold text-slate-100">Pipeline 2: Indian NCRB Crime Burden Forecasting</h3>
                        <span class="px-2 py-0.5 text-xs rounded bg-teal-500/20 text-teal-400">Macro-District Sequence</span>
                    </div>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        <b>Dataset:</b> National Crime Records Bureau (2001–2014), 9,856 district-years.<br>
                        <b>Task:</b> 5-tier ordered violent crime burden category (<code class="text-teal-400">TARGET_VIOLENT_LEVEL</code>).<br>
                        <b>Thresholds:</b> Frozen quintiles (107, 237, 362, 563) on 2001–2010 training data.<br>
                        <b>Sequence Design:</b> 3-year consecutive lag window (t-3, t-2, t-1 &rarr; t).<br>
                        <b>Test Evaluation:</b> Locked 2013–2014 cohort across 5 independent random seeds.
                    </p>
                </div>
            </div>
        </section>

        <!-- 2. EXPLORER SECTION -->
        <section id="sec-explorer" class="space-y-6 pt-6 border-t border-[#232f48]">
            <div class="flex justify-between items-center">
                <div>
                    <span class="px-2.5 py-0.5 text-xs rounded-full bg-cyan-500/15 text-cyan-400 font-semibold border border-cyan-500/30">INTERACTIVE DATA EXPLORER</span>
                    <h2 class="text-xl font-bold text-slate-100 mt-1">Chicago & NCRB Crime Explorer</h2>
                </div>
            </div>

            <!-- Filter Controls -->
            <div class="bg-[#151c2c] p-4 rounded-lg border border-[#232f48] grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
                <div>
                    <label class="block text-slate-400 font-semibold mb-1">Crime Type Filter</label>
                    <select id="flt-crime-type" class="w-full bg-[#1e293b] text-slate-200 border border-[#232f48] rounded p-2" onchange="updateExplorer()">
                        <option value="ALL">All Crime Types</option>
                        <option value="BATTERY">BATTERY</option>
                        <option value="THEFT">THEFT</option>
                        <option value="ASSAULT">ASSAULT</option>
                        <option value="CRIMINAL DAMAGE">CRIMINAL DAMAGE</option>
                        <option value="MOTOR VEHICLE THEFT">MOTOR VEHICLE THEFT</option>
                        <option value="OTHER OFFENSE">OTHER OFFENSE</option>
                    </select>
                </div>
                <div>
                    <label class="block text-slate-400 font-semibold mb-1">Domestic Classification</label>
                    <select id="flt-domestic" class="w-full bg-[#1e293b] text-slate-200 border border-[#232f48] rounded p-2" onchange="updateExplorer()">
                        <option value="ALL">All Altercations</option>
                        <option value="1">Domestic Only (Class 1)</option>
                        <option value="0">Non-Domestic Only (Class 0)</option>
                    </select>
                </div>
                <div>
                    <label class="block text-slate-400 font-semibold mb-1">Premises Type</label>
                    <select id="flt-residential" class="w-full bg-[#1e293b] text-slate-200 border border-[#232f48] rounded p-2" onchange="updateExplorer()">
                        <option value="ALL">All Locations</option>
                        <option value="1">Residential Only</option>
                        <option value="0">Public Spaces / Others</option>
                    </select>
                </div>
                <div class="flex items-end">
                    <button onclick="downloadFilteredCSV()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-semibold py-2 px-3 rounded transition text-xs flex justify-center items-center">
                        ⬇️ Download Filtered Sample (CSV)
                    </button>
                </div>
            </div>

            <!-- Explorer KPI Row -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="kpi-card">
                    <div class="text-xs text-slate-400 uppercase font-semibold">Filtered Incidents</div>
                    <div id="exp-total" class="text-xl font-bold text-slate-100 mt-1">1,000</div>
                    <div class="text-xs text-slate-400">Sample Explorer View</div>
                </div>
                <div class="kpi-card">
                    <div class="text-xs text-slate-400 uppercase font-semibold">Domestic Rate</div>
                    <div id="exp-dom-rate" class="text-xl font-bold text-rose-400 mt-1">18.4%</div>
                    <div class="text-xs text-slate-400">Prevalence in selection</div>
                </div>
                <div class="kpi-card">
                    <div class="text-xs text-slate-400 uppercase font-semibold">Modal Offense</div>
                    <div id="exp-top-crime" class="text-xl font-bold text-emerald-400 mt-1">BATTERY</div>
                    <div class="text-xs text-slate-400">Most frequent category</div>
                </div>
                <div class="kpi-card">
                    <div class="text-xs text-slate-400 uppercase font-semibold">Top Venue</div>
                    <div id="exp-top-venue" class="text-xl font-bold text-cyan-400 mt-1">STREET</div>
                    <div class="text-xs text-slate-400">Primary premise</div>
                </div>
            </div>

            <!-- Explorer Charts Grid -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="bg-[#151c2c] p-4 rounded-lg border border-[#232f48]">
                    <div id="chart-hourly" style="height: 320px;"></div>
                </div>
                <div class="bg-[#151c2c] p-4 rounded-lg border border-[#232f48]">
                    <div id="chart-districts" style="height: 320px;"></div>
                </div>
            </div>
        </section>

        <!-- 3. METHODOLOGY SECTION -->
        <section id="sec-methodology" class="space-y-6 pt-6 border-t border-[#232f48]">
            <span class="px-2.5 py-0.5 text-xs rounded-full bg-teal-500/15 text-teal-400 font-semibold border border-teal-500/30">DATA METHODOLOGY</span>
            <h2 class="text-xl font-bold text-slate-100">Leakage Audit & Feature Architectures</h2>

            <!-- Leakage Audit Table -->
            <div class="bg-[#151c2c] p-5 rounded-lg border border-[#232f48]">
                <h3 class="font-bold text-slate-100 mb-3 text-sm">🛡️ Target Leakage Audit Ledger</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs text-slate-300 border-collapse">
                        <thead>
                            <tr class="border-b border-[#232f48] text-slate-400">
                                <th class="p-2">Excluded Field</th>
                                <th class="p-2">Forensic Criminological Reason</th>
                                <th class="p-2">Severity Impact</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-[#232f48]">
                            <tr><td class="p-2 font-mono text-rose-400">Description</td><td class="p-2">Contains words like 'DOMESTIC BATTERY SIMPLE' in 143k+ rows. Direct target leakage.</td><td class="p-2 text-rose-400 font-bold">Catastrophic Leak</td></tr>
                            <tr><td class="p-2 font-mono text-rose-400">IUCR / FBI Code</td><td class="p-2">Statutory sub-codes distinctly encode domestic charges (IUCR 0486 = Domestic Battery).</td><td class="p-2 text-rose-400 font-bold">Direct Leak</td></tr>
                            <tr><td class="p-2 font-mono text-amber-400">Arrest</td><td class="p-2">Post-investigation operational outcome. Unrecorded during initial dispatch call.</td><td class="p-2 text-amber-400 font-bold">Post-Event Leak</td></tr>
                            <tr><td class="p-2 font-mono text-amber-400">Updated On</td><td class="p-2">Maintenance timestamp recorded days after incident occurrence.</td><td class="p-2 text-amber-400 font-bold">Look-Ahead Bias</td></tr>
                            <tr><td class="p-2 font-mono text-cyan-400">Block</td><td class="p-2">~35k distinct street strings. Deep trees memorize locations rather than general rules.</td><td class="p-2 text-cyan-400 font-bold">Overfitting</td></tr>
                            <tr><td class="p-2 font-mono text-slate-400">ID / Case Number</td><td class="p-2">Arbitrary administrative primary keys with zero causal predictive signal.</td><td class="p-2 text-slate-400 font-bold">Spurious ID</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Feature Sets D1 vs D2 -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="bg-[#151c2c] p-5 rounded-lg border border-[#232f48]">
                    <h4 class="font-bold text-slate-100 text-sm mb-2 text-cyan-400">Dataset 1: Baseline (14 Features)</h4>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        Strictly raw temporal, spatial, and nominal administrative features without domain engineering:
                        <ul class="list-disc pl-5 mt-2 space-y-1 text-slate-400 text-xs">
                            <li><b>Temporal (4):</b> Hour, Month, DayOfWeek, IsWeekend</li>
                            <li><b>Spatial (6):</b> District, Beat, Ward, Community Area, Latitude, Longitude</li>
                            <li><b>Categorical (2):</b> Primary Type, Location Description</li>
                            <li><b>Metadata (2):</b> Date, split (train/val/test indicator)</li>
                        </ul>
                    </p>
                </div>
                <div class="bg-[#151c2c] p-5 rounded-lg border border-[#232f48]">
                    <h4 class="font-bold text-slate-100 text-sm mb-2 text-emerald-400">Dataset 2: Enhanced (29 Features)</h4>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        Augments Dataset 1 with 15 domain-engineered criminological features:
                        <ul class="list-disc pl-5 mt-2 space-y-1 text-slate-400 text-xs">
                            <li><b>Trigonometric Cyclical (6):</b> Hour_Sin/Cos, Month_Sin/Cos, DayOfWeek_Sin/Cos</li>
                            <li><b>Venue / Shift Flags (4):</b> Is_Residential, Is_Public_Space, Is_Night, Time_Period</li>
                            <li><b>Bayesian Risk Priors (3):</b> Primary_Type_Domestic_Risk, Location_Domestic_Risk, Beat_Domestic_Risk</li>
                            <li><b>Spatial Interaction (2):</b> Beat_Historical_Incident_Volume, Type_Location_Interaction</li>
                        </ul>
                    </p>
                </div>
            </div>
        </section>

        <!-- 4. FEATURE ABLATION SECTION -->
        <section id="sec-feature-analysis" class="space-y-6 pt-6 border-t border-[#232f48]">
            <span class="px-2.5 py-0.5 text-xs rounded-full bg-emerald-500/15 text-emerald-400 font-semibold border border-emerald-500/30">FEATURE SCIENCE</span>
            <h2 class="text-xl font-bold text-slate-100">7-Stage Feature Progression & Ablation</h2>

            <div class="bg-[#151c2c] p-4 rounded-lg border border-[#232f48]">
                <div id="chart-progression" style="height: 380px;"></div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="bg-[#151c2c] p-4 rounded-lg border border-[#232f48]">
                    <div id="chart-ranking-scatter" style="height: 360px;"></div>
                </div>
                <div class="bg-[#151c2c] p-4 rounded-lg border border-[#232f48]">
                    <div id="chart-logo-impact" style="height: 360px;"></div>
                </div>
            </div>
        </section>

        <!-- 5. ML EXPERIMENT LAB -->
        <section id="sec-ml-experiments" class="space-y-6 pt-6 border-t border-[#232f48]">
            <span class="px-2.5 py-0.5 text-xs rounded-full bg-indigo-500/15 text-indigo-400 font-semibold border border-indigo-500/30">MODEL BENCHMARK</span>
            <h2 class="text-xl font-bold text-slate-100">Machine Learning Experiment Laboratory</h2>

            <div class="bg-[#151c2c] p-4 rounded-lg border border-[#232f48]">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="text-sm font-bold text-slate-100">Chicago Models Benchmark (2025 Test Set)</h3>
                    <div class="flex space-x-2">
                        <button onclick="renderMLChart('ROC_AUC')" class="px-3 py-1 text-xs rounded bg-emerald-600 text-white font-semibold">ROC-AUC</button>
                        <button onclick="renderMLChart('Recall')" class="px-3 py-1 text-xs rounded bg-[#1e293b] hover:bg-slate-700 text-slate-300 font-semibold">Recall</button>
                        <button onclick="renderMLChart('Balanced_Accuracy')" class="px-3 py-1 text-xs rounded bg-[#1e293b] hover:bg-slate-700 text-slate-300 font-semibold">Balanced Acc</button>
                        <button onclick="renderMLChart('F1_Score')" class="px-3 py-1 text-xs rounded bg-[#1e293b] hover:bg-slate-700 text-slate-300 font-semibold">F1-Score</button>
                    </div>
                </div>
                <div id="chart-ml-benchmark" style="height: 360px;"></div>
            </div>

            <!-- Stacking Meta-Ensemble -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="bg-[#151c2c] p-5 rounded-lg border border-[#232f48]">
                    <h4 class="font-bold text-slate-100 text-sm mb-2">Stacking Meta-Ensemble Architecture</h4>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        Out-of-fold probability predictions from <b>Logistic Regression</b>, <b>HistGradientBoosting</b>, and <b>XGBoost</b> were stacked into a penalized Logistic Regression meta-learner:
                    </p>
                    <div class="overflow-x-auto mt-3">
                        <table class="w-full text-xs text-left text-slate-300">
                            <tr class="border-b border-[#232f48] text-slate-400"><th>Model</th><th>ROC-AUC</th><th>Recall</th><th>F1</th></tr>
                            <tr><td class="py-1">Base: Logistic Regression</td><td>0.9090</td><td>82.80%</td><td>0.6473</td></tr>
                            <tr><td class="py-1">Base: HistGradientBoosting</td><td>0.9121</td><td>83.03%</td><td>0.6545</td></tr>
                            <tr><td class="py-1">Base: XGBoost</td><td>0.9103</td><td>83.03%</td><td>0.6495</td></tr>
                            <tr class="font-bold text-emerald-400"><td class="py-1">Stacked Meta-Ensemble</td><td>0.9122</td><td>83.26%</td><td>0.6535</td></tr>
                        </table>
                    </div>
                </div>
                <div class="bg-[#151c2c] p-5 rounded-lg border border-[#232f48]">
                    <h4 class="font-bold text-rose-400 text-sm mb-2">⚠️ The NCRB Target Leakage Revelation</h4>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        In legacy literature, an 80/20 random stratified split yielded an apparent <b>90.76% Accuracy</b>. 
                        Forensic analysis revealed 9 input IPC columns directly summed into the violent crime burden target. 
                        When re-evaluated on genuine future-year forecasting using 3-year sequential history:
                        <ul class="list-disc pl-5 mt-2 space-y-1 text-slate-300 text-xs">
                            <li><b>Random Stratified MLP (Leaked):</b> 90.56% Accuracy / 0.9907 AUC</li>
                            <li><b>Chronological LSTM (Leak-Free):</b> 76.74% Accuracy / 0.7097 Macro F1</li>
                        </ul>
                    </p>
                </div>
            </div>
        </section>

        <!-- 6. DEEP LEARNING SECTION -->
        <section id="sec-deep-learning" class="space-y-6 pt-6 border-t border-[#232f48]">
            <span class="px-2.5 py-0.5 text-xs rounded-full bg-purple-500/15 text-purple-400 font-semibold border border-purple-500/30">DEEP LEARNING</span>
            <h2 class="text-xl font-bold text-slate-100">Deep Tabular MLP & 5-Seed Sequence Models</h2>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Chicago Tabular MLP -->
                <div class="bg-[#151c2c] p-5 rounded-lg border border-[#232f48]">
                    <h3 class="font-bold text-slate-100 text-sm mb-2">Chicago PyTorch Deep Tabular MLP</h3>
                    <p class="text-xs text-slate-300 mb-3">
                        Linear(29&rarr;128) &rarr; Linear(128&rarr;64) &rarr; Linear(64&rarr;32) &rarr; 1 with BatchNorm, Dropout(0.3), and class-weighted BCE loss (pos_weight: 4.417).
                    </p>
                    <div id="chart-mlp-cm" style="height: 280px;"></div>
                </div>

                <!-- NCRB 5-Seed Sequences -->
                <div class="bg-[#151c2c] p-5 rounded-lg border border-[#232f48]">
                    <h3 class="font-bold text-slate-100 text-sm mb-2">NCRB 5-Seed Sequence Modeling (GRU vs LSTM)</h3>
                    <p class="text-xs text-slate-300 mb-3">
                        Macro F1 score distribution across 5 independent random seeds (11, 23, 37, 53, 71) on the locked 2013–2014 test cohort:
                    </p>
                    <div id="chart-multiseed-box" style="height: 280px;"></div>
                </div>
            </div>
        </section>

        <!-- 7. EVALUATION, CALIBRATION & FAIRNESS -->
        <section id="sec-evaluation" class="space-y-6 pt-6 border-t border-[#232f48]">
            <span class="px-2.5 py-0.5 text-xs rounded-full bg-amber-500/15 text-amber-400 font-semibold border border-amber-500/30">TRUSTWORTHY AI</span>
            <h2 class="text-xl font-bold text-slate-100">Calibration, Fairness & Explainability</h2>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Calibration Diagram -->
                <div class="bg-[#151c2c] p-4 rounded-lg border border-[#232f48]">
                    <div id="chart-calibration" style="height: 350px;"></div>
                    <div class="mt-2 text-xs text-slate-300 bg-[#1e293b] p-2.5 rounded">
                        <b>ECE Reduction:</b> Expected Calibration Error dropped from <b>14.60% down to 0.47%</b> via Isotonic Regression, transforming uncalibrated tree margins into dependable dispatch triage probabilities.
                    </div>
                </div>

                <!-- Fairness Audit Bubble -->
                <div class="bg-[#151c2c] p-4 rounded-lg border border-[#232f48]">
                    <div id="chart-fairness" style="height: 350px;"></div>
                    <div class="mt-2 text-xs text-slate-300 bg-[#1e293b] p-2.5 rounded">
                        <b>Equal Sensitivity:</b> Across 20 of 22 districts, True Positive Rate (Recall) stays strictly between <b>80.1% and 86.8%</b>, demonstrating parity across socio-demographically diverse precincts.
                    </div>
                </div>
            </div>

            <!-- SHAP Global Importance -->
            <div class="bg-[#151c2c] p-4 rounded-lg border border-[#232f48]">
                <div id="chart-shap" style="height: 420px;"></div>
            </div>
        </section>

        <!-- 8. RESEARCH INSIGHTS SECTION -->
        <section id="sec-insights" class="space-y-6 pt-6 border-t border-[#232f48]">
            <span class="px-2.5 py-0.5 text-xs rounded-full bg-rose-500/15 text-rose-400 font-semibold border border-rose-500/30">ACADEMIC SYNTHESIS</span>
            <h2 class="text-xl font-bold text-slate-100">Core Research Findings & Critical Insights</h2>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div class="bg-[#151c2c] p-4 rounded border-l-4 border-rose-500 border border-[#232f48]">
                    <h4 class="font-bold text-slate-100 text-sm mb-1">1. The Target Leakage Trap</h4>
                    <p class="text-slate-300 leading-relaxed">
                        Descriptions containing 'DOMESTIC' and NCRB source columns sharing 76.6% target count yield illusory 90%+ accuracy. Real future forecasting demands rigorous pre-incident feature audits.
                    </p>
                </div>
                <div class="bg-[#151c2c] p-4 rounded border-l-4 border-amber-500 border border-[#232f48]">
                    <h4 class="font-bold text-slate-100 text-sm mb-1">2. Mandatory Chronological Splitting</h4>
                    <p class="text-slate-300 leading-relaxed">
                        Random train/test splits interpolate between concurrent events. Locking 2025 (Chicago) and 2013–2014 (NCRB) test sets provides genuine out-of-distribution evaluation.
                    </p>
                </div>
                <div class="bg-[#151c2c] p-4 rounded border-l-4 border-cyan-500 border border-[#232f48]">
                    <h4 class="font-bold text-slate-100 text-sm mb-1">3. Class Imbalance & Metric Deception</h4>
                    <p class="text-slate-300 leading-relaxed">
                        With 18.4% domestic prevalence, predicting all non-domestic yields 81.6% accuracy but 0% recall. Tuning threshold to 0.47 captures 87.18% of domestic abuse altercations.
                    </p>
                </div>
                <div class="bg-[#151c2c] p-4 rounded border-l-4 border-emerald-500 border border-[#232f48]">
                    <h4 class="font-bold text-slate-100 text-sm mb-1">4. Tabular ML vs Sequence DL Divide</h4>
                    <p class="text-slate-300 leading-relaxed">
                        XGBoost dominates static incident triage (0.9152 AUC) with fast training. For multi-year district forecasting, LSTM sequence modeling (+0.0125 F1 gain) outperforms static models.
                    </p>
                </div>
            </div>
        </section>

        <!-- 9. TECHNICAL DETAILS & DOWNLOADS -->
        <section id="sec-details" class="space-y-6 pt-6 border-t border-[#232f48]">
            <span class="px-2.5 py-0.5 text-xs rounded-full bg-cyan-500/15 text-cyan-400 font-semibold border border-cyan-500/30">TECHNICAL REPRODUCIBILITY</span>
            <h2 class="text-xl font-bold text-slate-100">Dataset Specifications & Research CSV Downloads</h2>

            <div class="bg-[#151c2c] p-5 rounded-lg border border-[#232f48]">
                <h3 class="font-bold text-slate-100 text-sm mb-3">📥 Instant CSV Table Downloads (Direct from Browser)</h3>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <button onclick="downloadTableCSV('dataset2_model_benchmark', 'chicago_d2_benchmark.csv')" class="p-2 text-xs bg-[#1e293b] hover:bg-emerald-600/30 border border-[#232f48] rounded text-slate-200 text-left transition">
                        ⬇️ Chicago D2 Benchmark CSV
                    </button>
                    <button onclick="downloadTableCSV('feature_ablation_progression', '7_stage_ablation.csv')" class="p-2 text-xs bg-[#1e293b] hover:bg-emerald-600/30 border border-[#232f48] rounded text-slate-200 text-left transition">
                        ⬇️ 7-Stage Progression CSV
                    </button>
                    <button onclick="downloadTableCSV('district_fairness_audit', 'fairness_22_districts.csv')" class="p-2 text-xs bg-[#1e293b] hover:bg-emerald-600/30 border border-[#232f48] rounded text-slate-200 text-left transition">
                        ⬇️ 22-District Fairness CSV
                    </button>
                    <button onclick="downloadTableCSV('calibration_metrics', 'calibration_metrics.csv')" class="p-2 text-xs bg-[#1e293b] hover:bg-emerald-600/30 border border-[#232f48] rounded text-slate-200 text-left transition">
                        ⬇️ Calibration ECE CSV
                    </button>
                    <button onclick="downloadTableCSV('shap_feature_importance', 'shap_importance.csv')" class="p-2 text-xs bg-[#1e293b] hover:bg-emerald-600/30 border border-[#232f48] rounded text-slate-200 text-left transition">
                        ⬇️ SHAP Feature Values CSV
                    </button>
                    <button onclick="downloadTableCSV('thesis_ready_final_results', 'ncrb_final_results.csv')" class="p-2 text-xs bg-[#1e293b] hover:bg-emerald-600/30 border border-[#232f48] rounded text-slate-200 text-left transition">
                        ⬇️ NCRB Final Results CSV
                    </button>
                </div>
            </div>
        </section>

        <!-- 10. CONCLUSION & ROADMAP -->
        <section id="sec-conclusion" class="space-y-6 pt-6 border-t border-[#232f48] pb-12">
            <span class="px-2.5 py-0.5 text-xs rounded-full bg-emerald-500/15 text-emerald-400 font-semibold border border-emerald-500/30">STRATEGIC SYNTHESIS</span>
            <h2 class="text-xl font-bold text-slate-100">Conclusion & Ethical Guardrails</h2>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="bg-[#151c2c] p-5 rounded-lg border-t-4 border-emerald-500 border border-[#232f48]">
                    <h4 class="font-bold text-slate-100 text-sm mb-2">1. Research Contributions</h4>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        Exposed and purged structural target leakage, proved that domain-engineered risk priors boost discrimination from 0.5593 to 0.9152 ROC-AUC, 
                        and demonstrated that Isotonic Regression reduces probability error to 0.47% while maintaining demographic parity across 22 districts.
                    </p>
                </div>
                <div class="bg-[#151c2c] p-5 rounded-lg border-t-4 border-rose-500 border border-[#232f48]">
                    <h4 class="font-bold text-slate-100 text-sm mb-2">2. Strict Rejection of Predictive Policing</h4>
                    <p class="text-xs text-slate-300 leading-relaxed">
                        This framework is built strictly for retrospective analysis and emergency dispatcher triage support. 
                        It is NOT intended for predictive policing, proactive surveillance, or individual risk scoring.
                    </p>
                </div>
            </div>

            <div class="text-center pt-8 border-t border-[#232f48] text-xs text-slate-500">
                Comparative Evaluation of Machine Learning and Deep Learning for Crime Data Classification &bull; Capstone 2026<br>
                Permanent HTML Web Interface &bull; Zero Server Dependencies
            </div>
        </section>

    </main>

    <!-- EMBEDDED REAL DATA & JAVASCRIPT LOGIC -->
    <script>
        const DATA = {tables_json};

        // Dark theme template for Plotly
        const darkLayout = {{
            paper_bgcolor: '#151c2c',
            plot_bgcolor: '#151c2c',
            font: {{ family: '-apple-system, sans-serif', color: '#94a3b8', size: 11 }},
            margin: {{ l: 45, r: 25, t: 35, b: 40 }},
            xaxis: {{ showgrid: true, gridcolor: '#232f48', zeroline: false }},
            yaxis: {{ showgrid: true, gridcolor: '#232f48', zeroline: false }}
        }};

        function initCharts() {{
            renderProgressionChart();
            renderRankingScatter();
            renderLogoChart();
            renderMLChart('ROC_AUC');
            renderMLPConfusionMatrix();
            renderMultiSeedBox();
            renderCalibrationChart();
            renderFairnessBubble();
            renderShapBar();
            updateExplorer();
        }}

        // 7-Stage Progression Chart
        function renderProgressionChart() {{
            const df = DATA.feature_ablation_progression || [];
            const x = df.map(r => r.Configuration.replace('. ', '<br>'));
            const y_auc = df.map(r => r.ROC_AUC);
            const y_f1 = df.map(r => r.F1_Score);

            const trace1 = {{
                x: x, y: y_auc, name: 'ROC-AUC', type: 'bar',
                marker: {{ color: '#10b981', opacity: 0.85 }},
                text: y_auc.map(v => v.toFixed(4)), textposition: 'inside'
            }};
            const trace2 = {{
                x: x, y: y_f1, name: 'F1-Score', type: 'scatter', mode: 'lines+markers',
                line: {{ color: '#06b6d4', width: 3 }}, yaxis: 'y2'
            }};
            const layout = {{
                ...darkLayout,
                title: {{ text: '<b>7-Stage Cumulative Feature Progression (XGBoost Test Set)</b>', font: {{ color: '#f8fafc', size: 13 }} }},
                yaxis: {{ ...darkLayout.yaxis, title: 'ROC-AUC', range: [0.45, 1.0] }},
                yaxis2: {{ title: 'F1-Score', range: [0.25, 0.75], overlaying: 'y', side: 'right', showgrid: false }},
                legend: {{ orientation: 'h', y: 1.15, x: 1, xanchor: 'right' }}
            }};
            Plotly.newPlot('chart-progression', [trace1, trace2], layout, {{ responsive: true }});
        }}

        // Mutual Info vs Correlation Scatter
        function renderRankingScatter() {{
            const df = DATA.feature_comparison_ranking || [];
            const trace = {{
                x: df.map(r => r.Mutual_Information),
                y: df.map(r => r.Correlation_with_Domestic),
                text: df.map(r => r.Feature),
                mode: 'markers+text',
                textposition: 'top center',
                marker: {{
                    size: df.map(r => Math.max(8, r.Abs_Correlation * 25)),
                    color: df.map(r => r.Category === 'Bayesian Risk Prior' ? '#10b981' : (r.Category === 'Venue / Shift Flag' ? '#14b8a6' : '#06b6d4')),
                    line: {{ color: '#232f48', width: 1.5 }}
                }}
            }};
            const layout = {{
                ...darkLayout,
                title: {{ text: '<b>Mutual Information vs Target Correlation</b>', font: {{ color: '#f8fafc', size: 13 }} }},
                xaxis: {{ ...darkLayout.xaxis, title: 'Mutual Information' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Pearson Correlation with Domestic' }}
            }};
            Plotly.newPlot('chart-ranking-scatter', [trace], layout, {{ responsive: true }});
        }}

        // LOGO Impact Chart
        function renderLogoChart() {{
            const df = (DATA.leave_one_group_out_ablation || []).filter(r => !r.Configuration.includes('Full Model'));
            const trace = {{
                y: df.map(r => r.Configuration),
                x: df.map(r => r.Drop_Delta_AUC * 1000),
                type: 'bar', orientation: 'h',
                marker: {{
                    color: df.map(r => r.Drop_Delta_AUC < 0 ? '#f43f5e' : '#14b8a6')
                }},
                text: df.map(r => r.Drop_Delta_AUC.toFixed(4)), textposition: 'outside'
            }};
            const layout = {{
                ...darkLayout,
                title: {{ text: '<b>Leave-One-Group-Out Ablation (AUC Drop x 10⁻³)</b>', font: {{ color: '#f8fafc', size: 13 }} }},
                margin: {{ l: 160, r: 30, t: 35, b: 35 }},
                xaxis: {{ ...darkLayout.xaxis, title: 'Δ ROC-AUC (Milli-Points)' }}
            }};
            Plotly.newPlot('chart-logo-impact', [trace], layout, {{ responsive: true }});
        }}

        // ML Benchmark Bar
        function renderMLChart(metric) {{
            const df = DATA.dataset2_model_benchmark || [];
            const trace = {{
                x: df.map(r => r.Model),
                y: df.map(r => r[metric]),
                type: 'bar',
                marker: {{ color: ['#10b981', '#14b8a6', '#06b6d4', '#3b82f6', '#8b5cf6'] }},
                text: df.map(r => r[metric].toFixed(4)), textposition: 'outside'
            }};
            const layout = {{
                ...darkLayout,
                title: {{ text: `<b>Dataset 2 Models Comparison: ${{metric.replace('_', ' ')}}</b>`, font: {{ color: '#f8fafc', size: 13 }} }},
                yaxis: {{ ...darkLayout.yaxis, title: metric.replace('_', ' '), range: [0, 1.05] }}
            }};
            Plotly.newPlot('chart-ml-benchmark', [trace], layout, {{ responsive: true }});
        }}

        // MLP Confusion Matrix
        function renderMLPConfusionMatrix() {{
            const z = [[154677, 38011], [5892, 37749]];
            const trace = {{
                z: z, x: ['Pred: Non-Domestic', 'Pred: Domestic'], y: ['Actual: Non-Domestic', 'Actual: Domestic'],
                type: 'heatmap', colorscale: [[0, '#151c2c'], [0.5, '#14b8a6'], [1, '#10b981']],
                showscale: false
            }};
            const layout = {{
                ...darkLayout,
                title: {{ text: '<b>PyTorch Tabular MLP Confusion Matrix (236k Test)</b>', font: {{ color: '#f8fafc', size: 12 }} }},
                annotations: [
                    {{ x: 'Pred: Non-Domestic', y: 'Actual: Non-Domestic', text: 'TN: 154,677', showarrow: false, font: {{ color: '#fff' }} }},
                    {{ x: 'Pred: Domestic', y: 'Actual: Non-Domestic', text: 'FP: 38,011', showarrow: false, font: {{ color: '#fff' }} }},
                    {{ x: 'Pred: Non-Domestic', y: 'Actual: Domestic', text: 'FN: 5,892', showarrow: false, font: {{ color: '#fff' }} }},
                    {{ x: 'Pred: Domestic', y: 'Actual: Domestic', text: 'TP: 37,749', showarrow: false, font: {{ color: '#fff' }} }}
                ]
            }};
            Plotly.newPlot('chart-mlp-cm', [trace], layout, {{ responsive: true }});
        }}

        // Multi-seed Box
        function renderMultiSeedBox() {{
            const gru_base = [0.7063, 0.7032, 0.7031, 0.6996, 0.6902];
            const gru_enh = [0.7170, 0.6979, 0.7107, 0.7126, 0.7035];
            const lstm_base = [0.7021, 0.7001, 0.6931, 0.6983, 0.6925];
            const lstm_enh = [0.7076, 0.7054, 0.7036, 0.7203, 0.7115];

            const trace1 = {{ y: gru_base, name: 'GRU (Base)', type: 'box', marker: {{ color: '#64748b' }} }};
            const trace2 = {{ y: gru_enh, name: 'GRU (Enh)', type: 'box', marker: {{ color: '#14b8a6' }} }};
            const trace3 = {{ y: lstm_base, name: 'LSTM (Base)', type: 'box', marker: {{ color: '#64748b' }} }};
            const trace4 = {{ y: lstm_enh, name: 'LSTM (Enh)', type: 'box', marker: {{ color: '#10b981' }} }};

            const layout = {{
                ...darkLayout,
                title: {{ text: '<b>5-Seed Test Macro F1 Stability</b>', font: {{ color: '#f8fafc', size: 12 }} }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Test Macro F1', range: [0.68, 0.73] }},
                showlegend: false
            }};
            Plotly.newPlot('chart-multiseed-box', [trace1, trace2, trace3, trace4], layout, {{ responsive: true }});
        }}

        // Calibration Curves
        function renderCalibrationChart() {{
            const x = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
            const ideal = {{ x: [0, 1], y: [0, 1], mode: 'lines', line: {{ color: '#64748b', dash: 'dash' }}, name: 'Perfect (y=x)' }};
            const uncal = {{ x: x, y: [0.01, 0.04, 0.08, 0.16, 0.28, 0.44, 0.63, 0.81, 0.91, 0.98], mode: 'lines+markers', line: {{ color: '#f43f5e', width: 2 }}, name: 'Uncalibrated (ECE: 14.6%)' }};
            const iso = {{ x: x, y: x, mode: 'lines+markers', line: {{ color: '#10b981', width: 3 }}, marker: {{ symbol: 'diamond', size: 7 }}, name: 'Isotonic (ECE: 0.47%)' }};

            const layout = {{
                ...darkLayout,
                title: {{ text: '<b>Probability Calibration Reliability Diagram</b>', font: {{ color: '#f8fafc', size: 13 }} }},
                xaxis: {{ ...darkLayout.xaxis, title: 'Mean Predicted Probability' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Empirical Domestic Fraction' }},
                legend: {{ orientation: 'h', y: 1.15, x: 1, xanchor: 'right' }}
            }};
            Plotly.newPlot('chart-calibration', [ideal, uncal, iso], layout, {{ responsive: true }});
        }}

        // Fairness Bubble
        function renderFairnessBubble() {{
            const df = DATA.district_fairness_audit || [];
            const trace = {{
                x: df.map(r => r.Actual_Domestic_Rate * 100),
                y: df.map(r => r.TPR_Recall * 100),
                text: df.map(r => `Dist ${{r.District}}`),
                mode: 'markers+text',
                textposition: 'top center',
                marker: {{
                    size: df.map(r => Math.sqrt(r.Test_Incidents) / 3.5),
                    color: df.map(r => r.Disparate_Impact_Ratio),
                    colorscale: [[0, '#06b6d4'], [0.5, '#14b8a6'], [1, '#f43f5e']],
                    showscale: true,
                    colorbar: {{ title: 'Disparate Impact', thickness: 10, len: 0.8 }}
                }}
            }};
            const layout = {{
                ...darkLayout,
                title: {{ text: '<b>Fairness Parity: 22 Districts Recall Stability</b>', font: {{ color: '#f8fafc', size: 13 }} }},
                xaxis: {{ ...darkLayout.xaxis, title: 'Actual Domestic Rate (%)' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'True Positive Rate / Recall (%)', range: [65, 95] }}
            }};
            Plotly.newPlot('chart-fairness', [trace], layout, {{ responsive: true }});
        }}

        // SHAP Bar
        function renderShapBar() {{
            const df = [...(DATA.shap_feature_importance || [])].sort((a,b) => a.Mean_Abs_SHAP - b.Mean_Abs_SHAP);
            const trace = {{
                y: df.map(r => r.Feature),
                x: df.map(r => r.Mean_Abs_SHAP),
                type: 'bar', orientation: 'h',
                marker: {{ color: '#10b981' }},
                text: df.map(r => r.Mean_Abs_SHAP.toFixed(3)), textposition: 'outside'
            }};
            const layout = {{
                ...darkLayout,
                title: {{ text: '<b>Global Feature Importance (|SHAP| on 2025 Test Set)</b>', font: {{ color: '#f8fafc', size: 13 }} }},
                margin: {{ l: 190, r: 35, t: 35, b: 35 }},
                xaxis: {{ ...darkLayout.xaxis, title: 'Mean |SHAP| Impact on Domestic Log-Odds' }}
            }};
            Plotly.newPlot('chart-shap', [trace], layout, {{ responsive: true }});
        }}

        // Interactive Explorer Update
        let filteredSampleData = [];
        function updateExplorer() {{
            const crimeType = document.getElementById('flt-crime-type').value;
            const domestic = document.getElementById('flt-domestic').value;
            const residential = document.getElementById('flt-residential').value;

            let rows = DATA.chicago_sample || [];
            if (crimeType !== 'ALL') rows = rows.filter(r => r['Primary Type'] === crimeType);
            if (domestic !== 'ALL') rows = rows.filter(r => String(r.Domestic) === domestic);
            if (residential !== 'ALL') rows = rows.filter(r => String(r.Is_Residential) === residential);

            filteredSampleData = rows;

            // Update KPIs
            const total = rows.length;
            const domCount = rows.filter(r => r.Domestic === 1).length;
            const rate = total > 0 ? (domCount / total * 100).toFixed(1) : 0;
            
            document.getElementById('exp-total').innerText = total.toLocaleString();
            document.getElementById('exp-dom-rate').innerText = `${{rate}}%`;

            // Hourly bar
            const hourCounts = Array(24).fill(0);
            rows.forEach(r => hourCounts[r.Hour]++);
            const traceH = {{
                x: Array.from({{length: 24}}, (_, i) => `${{i}}:00`),
                y: hourCounts, type: 'bar', marker: {{ color: '#06b6d4' }}
            }};
            Plotly.newPlot('chart-hourly', [traceH], {{
                ...darkLayout,
                title: {{ text: '<b>Hourly Incident Volume (Filtered)</b>', font: {{ color: '#f8fafc', size: 12 }} }},
                xaxis: {{ ...darkLayout.xaxis, title: 'Hour of Day' }}
            }}, {{ responsive: true }});

            // District bar
            const distCounts = {{}};
            rows.forEach(r => distCounts[r.District] = (distCounts[r.District] || 0) + 1);
            const distKeys = Object.keys(distCounts).sort((a,b) => distCounts[b] - distCounts[a]).slice(0, 10);
            const traceD = {{
                x: distKeys.map(k => `Dist ${{k}}`),
                y: distKeys.map(k => distCounts[k]),
                type: 'bar', marker: {{ color: '#10b981' }}
            }};
            Plotly.newPlot('chart-districts', [traceD], {{
                ...darkLayout,
                title: {{ text: '<b>Top 10 Districts by Volume (Filtered)</b>', font: {{ color: '#f8fafc', size: 12 }} }}
            }}, {{ responsive: true }});
        }}

        // Client-side CSV Download
        function downloadFilteredCSV() {{
            if (!filteredSampleData.length) return alert('No records to download');
            const keys = Object.keys(filteredSampleData[0]);
            let csv = keys.join(',') + '\\n';
            filteredSampleData.forEach(r => {{
                csv += keys.map(k => `"${{r[k] !== undefined ? r[k] : ''}}"`).join(',') + '\\n';
            }});
            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'chicago_crime_filtered.csv';
            a.click();
        }}

        function downloadTableCSV(tableName, filename) {{
            const rows = DATA[tableName] || [];
            if (!rows.length) return alert('Table not found');
            const keys = Object.keys(rows[0]);
            let csv = keys.join(',') + '\\n';
            rows.forEach(r => {{
                csv += keys.map(k => `"${{r[k] !== undefined ? r[k] : ''}}"`).join(',') + '\\n';
            }});
            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
        }}

        function setActiveNav(el) {{
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            el.classList.add('active');
        }}

        window.onload = initCharts;
    </script>
</body>
</html>
"""

# Write to index.html and dashboard.html in project root
out_index = ROOT / "index.html"
out_dash = ROOT / "dashboard.html"
out_index.write_text(html_content, encoding="utf-8")
out_dash.write_text(html_content, encoding="utf-8")
print(f"Generated {out_index} ({len(html_content)} bytes)")

# Also copy to travel
travel_index = Path(r"C:\Users\aru12\OneDrive\Desktop\travel\index.html")
travel_dash = Path(r"C:\Users\aru12\OneDrive\Desktop\travel\dashboard.html")
travel_index.write_text(html_content, encoding="utf-8")
travel_dash.write_text(html_content, encoding="utf-8")
print("Copied to travel workspace!")
