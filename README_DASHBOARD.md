# Interactive Crime Analytics & Machine Learning Research Dashboard

## 📌 Project Overview
**Title:** Comparative Evaluation of Machine Learning and Deep Learning for Crime Data Classification  
**Purpose:** An interactive, peer-review compliant web dashboard built with **Streamlit** and **Plotly** to showcase, explore, and evaluate the complete dual-pipeline research study.

---

## 🚀 Quick Start: How to Run the Dashboard

### Option 1: Double-Click Batch File (Easiest)
Simply double-click the file:
`run_dashboard.bat` in the project directory.

### Option 2: PowerShell / Command Prompt
Open your terminal in this folder:
```powershell
cd "C:\Users\aru12\OneDrive\Desktop\capstone project"
& "C:\Users\aru12\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🏛️ Architecture & Sections Implemented

The application is structured into 10 modular, research-oriented command center sections:

1. **🏛️ Overview & Research at a Glance (`modules/overview.py`):**
   - Research objectives and dual-pipeline framing.
   - High-level empirical KPI cards (765.7k records, 0.9152 ROC-AUC, 87.18% Recall, 0.47% ECE).
   - Interactive Mermaid methodology flowchart and core research principles.

2. **🔍 Crime Data Explorer (`modules/explorer.py`):**
   - **Chicago Police Incident Explorer:** Filter by Year (2023–2025), Crime Category, Police District (1–25), and Domestic Altercation flag. Dynamic hourly, weekly, district-level, and venue-level charts.
   - **Indian NCRB District Panel:** Filter by State, Year Range (2001–2014), and Violent Burden Tier (1 to 5). Visual trajectories of 6 core crime groups, Shannon entropy diversity index, and state-level burden distributions.
   - Export filtered subsets as CSV.

3. **⚙️ Data Processing & Methodology (`modules/methodology.py`):**
   - Detailed leakage audit ledger (why `Description`, `IUCR`, `Arrest`, and `Updated On` were excluded).
   - Dataset 1 (14 Baseline Features) vs Dataset 2 (29 Enhanced Features) architecture.
   - NCRB consolidation into 6 crime families and frozen quintile burden boundaries (107, 237, 362, 563).
   - Strict 3-year sliding lag sequence specification ($t-3, t-2, t-1 \to t$).

4. **🔬 Feature Engineering & Ablation (`modules/feature_analysis.py`):**
   - 7-Stage Cumulative Progression chart (ROC-AUC rising from 0.5593 to 0.9150).
   - Information Theoretic (Mutual Information) vs Linear (Pearson Correlation) quadrant chart.
   - Leave-One-Group-Out (LOGO) ablation analysis.
   - NCRB 5-seed paired feature ablation (evaluating marginal contributions of dynamic indicators).

5. **🧪 Machine Learning Experiment Lab (`modules/ml_experiments.py`):**
   - Interactive model benchmark comparing Logistic Regression, Random Forest, HistGradientBoosting, XGBoost (Champion), Deep Tabular MLP, and Stacking Meta-Ensemble.
   - Multi-metric filtering (ROC-AUC, Balanced Accuracy, Recall, F1, Precision, Accuracy).
   - Threshold tuning analysis (why 0.47 was selected to prioritize 87.18% Recall).
   - NCRB chronological sequence benchmark (GRU, LSTM, Logistic Regression, Majority Baseline) vs legacy target-leaked models.

6. **🧠 Deep Learning Experiments (`modules/deep_learning.py`):**
   - Chicago Deep Tabular MLP: 4-layer PyTorch neural network with BatchNorm, Dropout, and inverse-frequency weighted cross-entropy. Confusion matrix on 236k test incidents.
   - NCRB Recurrent Sequence Models: Gated Recurrent Unit (GRU) vs Long Short-Term Memory (LSTM) across 5 independent random seeds (11, 23, 37, 53, 71) with mean $\pm$ standard deviation stability box plots.

7. **📊 Evaluation, Calibration & Fairness (`modules/evaluation.py`):**
   - **Probability Calibration:** Reliability diagram showing Expected Calibration Error (ECE) reduction from 14.60% down to 0.47% via Isotonic Regression (Brier Score: 0.0867).
   - **Algorithmic Fairness Audit:** Comprehensive audit across 22 Chicago Police Districts on the 2025 Test Set demonstrating stable True Positive Rates (80%–87%) and analyzing Disparate Impact ratios.
   - **SHAP Explainability:** TreeSHAP global feature importance ranking identifying Bayesian risk priors and residential venue flags as primary decision drivers.

8. **💡 Research Findings & Insights (`modules/research_insights.py`):**
   - Academic breakdown of The Data Leakage Trap, Chronological Validation vs Random Splits, Class Imbalance Reality, ML vs DL Trade-offs, and Ethical Boundaries.

9. **📋 Dataset & Experiment Details (`modules/dataset_details.py`):**
   - Technical schemas, data dictionaries, model hyperparameters, random seeds, and downloadable CSVs of all experimental tables.

10. **🎯 Academic Conclusion & Roadmap (`modules/conclusion.py`):**
    - Synthesis of research achievements, operational dispatch recommendations, ethical caveats (rejection of predictive policing), and future roadmap (Spatio-Temporal GNNs, Multimodal 911 Audio).

---

## 📊 Summary of Verified Empirical Results

### Pipeline 1: Chicago Incident Classification (Domestic Violence Triage)
- **Dataset Size:** 756,085 records (2023–2025) across 22 active districts.
- **Class Prevalence:** 18.46% domestic violence incidents.
- **Champion Model:** XGBoost on Dataset 2 (29 features).
- **Test Set Performance (2025, 236,329 incidents):**
  - **ROC-AUC:** `0.9152`
  - **Balanced Accuracy:** `83.55%`
  - **Recall (Sensitivity):** `87.18%` (at decision threshold `0.47`)
  - **F1-Score:** `0.6406` (Precision: `50.63%`)
  - **Expected Calibration Error (ECE):** Reduced from `14.60%` to `0.47%` (Isotonic Regression).
  - **District Fairness:** True Positive Rate stays between `80.1%` and `86.8%` across 20 of 22 districts.

### Pipeline 2: Indian NCRB Burden Forecasting (3-Year Sequential History)
- **Dataset Size:** 9,856 district-years (2001–2014) across 35 States / UTs.
- **Target:** 5-tier ordered violent crime burden (frozen quintile cutoffs: 107, 237, 362, 563).
- **Champion Architecture:** Long Short-Term Memory (LSTM) with 5 engineered dynamic features.
- **Test Set Performance (2013–2014, 671 shared sequences):**
  - **Macro F1:** `0.7097` (LSTM Enhanced, 5-seed mean) vs `0.7041` (Baseline) and `0.6761` (Logistic Regression).
  - **Accuracy:** `72.73%` / `76.74%` (chronological sequence).
  - **Target Leakage Note:** Legacy 80/20 random stratified split yielded an illusory `90.76%` accuracy due to 9 overlapping IPC source variables. Leak-free chronological sequence modeling yields genuine operational performance.

---

## 🛠️ Technology Stack
- **Framework:** Streamlit
- **Visualizations:** Plotly Dark (`plotly.graph_objects`, `plotly.express`)
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, PyTorch
