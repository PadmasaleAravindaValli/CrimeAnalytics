# Computational Analysis of Crime and Social Systems: Machine Learning & Deep Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-185a9d.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Dashboard-brightgreen.svg)](https://padmasalearavindavalli.github.io/CrimeAnalytics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computational crime research platform and comparative machine learning / deep learning study evaluating incident-level domestic crime classification and spatio-temporal crime burden forecasting across 756,085 real-world police records (2023–2025).

---

## 🌐 Live Interactive Research Dashboard

The complete empirical research findings, interactive multi-attribute crime data explorer, SHAP explainability analyses, probability calibration curves, and district fairness audits are deployed live online:

* **Permanent Web URL:** **[padmasalearavindavalli.github.io/CrimeAnalytics](https://padmasalearavindavalli.github.io/CrimeAnalytics/)** *(Zero-code, accessible worldwide on mobile, tablet, and PC)*
* **Permanent Static QR Code:**
  
  <p align="center">
    <img src="permanent_dashboard_qr.png" alt="Permanent Dashboard QR Code" width="220" />
    <br>
    <em>Scan with any smartphone camera to launch the interactive research dashboard</em>
  </p>

* **Local Command Center:** Launch locally with one click via `run_dashboard.bat` (serves on `http://localhost:8501`).
* **Offline Portable View:** Double-click `dashboard.html` to run the fully responsive interactive dashboard locally without an internet connection.

---

## 📋 Table of Contents

- [Executive Summary & Dual-Pipeline Architecture](#-executive-summary--dual-pipeline-architecture)
- [Methodological Rigor & Target Leakage Purge](#-methodological-rigor--target-leakage-purge)
- [Key Empirical Results & Benchmarks](#-key-empirical-results--benchmarks)
  - [1. Incident Classification (Locked 2025 Test Cohort)](#1-incident-classification-locked-2025-test-cohort)
  - [2. 7-Stage Feature Ablation Progression](#2-7-stage-feature-ablation-progression)
  - [3. Isotonic Probability Calibration](#3-isotonic-probability-calibration)
  - [4. Spatio-Temporal Sequence Forecasting](#4-spatio-temporal-sequence-forecasting)
- [Feature Engineering: Dataset 1 vs Dataset 2](#-feature-engineering-dataset-1-vs-dataset-2)
- [Algorithmic Fairness Audit & TreeSHAP Explainability](#-algorithmic-fairness-audit--treeshap-explainability)
- [Streamlit Command Center Architecture](#-streamlit-command-center-architecture)
- [Repository File Structure](#-repository-file-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Ethical Guardrails & Public Policy Governance](#-ethical-guardrails--public-policy-governance)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🔬 Executive Summary & Dual-Pipeline Architecture

This project investigates domestic incident classification and spatio-temporal violent crime burden forecasting using rigorous machine learning and deep learning architectures. It addresses key methodological pitfalls in computational criminology, including target leakage, optimistic random splits, uncalibrated risk probabilities, and disparate algorithmic impact.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 CRIME ANALYTICS ARCHITECTURE                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌─────────────────────────────────────────┐         ┌─────────────────────────────────────────┐
│ PIPELINE 1: INCIDENT CLASSIFICATION     │         │ PIPELINE 2: SPATIO-TEMPORAL FORECASTING │
├─────────────────────────────────────────┤         ├─────────────────────────────────────────┤
│ • 756,085 CPD Incidents (2023–2025)     │         │ • 274 Police Beats across 36 Months     │
│ • 236,329 Locked 2025 Test Incidents    │         │ • 9,900 Beat-Month Sequential Panels    │
│ • Target: Domestic (Binary, 18.5%)      │         │ • 5-Tier Ordered Violent Burden Target  │
│ • Purged: Description, Arrest, IUCR     │         │ • Input History: [t-3, t-2, t-1] -> t   │
│ • Models: XGBoost, MLP, RF, HGB, LR     │         │ • Models: GRU, LSTM, Lag-1 Baseline     │
│ • Champion: XGBoost (0.9152 ROC-AUC)    │         │ • Champion: GRU (0.5111 Macro F1)       │
└─────────────────────────────────────────┘         └─────────────────────────────────────────┘
         │                                                                   │
         └─────────────────────────────────┬─────────────────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        COMPREHENSIVE GOVERNANCE & EVALUATION                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  ✓ 7-Stage Feature Ablation Progression (0.5593 -> 0.9150 ROC-AUC)                     │
│  ✓ Isotonic Probability Calibration (ECE reduced from 14.60% to 0.47%)                 │
│  ✓ Algorithmic Fairness Audit across 22 Police Districts (TPR Parity: 80.1% – 86.8%)   │
│  ✓ Global TreeSHAP Importance & Local Feature Risk Attributions                        │
│  ✓ Permutation Shuffled-Label Sanity Check (ROC-AUC drops to 0.5319)                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Methodological Rigor & Target Leakage Purge

A major contribution of this work is the identification and elimination of critical forms of **target leakage** that artificially inflate reported accuracy in published literature:

1. **Target Leakage Purge**:
   - `Description`: The raw CPD incident description field contained the exact token `"DOMESTIC"` in 143,000+ records (e.g., `"DOMESTIC BATTERY SIMPLE"`). Retaining this column results in trivial, non-generalizable 99%+ accuracy. It was completely stripped prior to modeling.
   - `Arrest`: Post-incident outcome determined hours after police arrival; cannot legitimately be known at dispatch time. Excluded.
   - `IUCR` & `Updated On`: Administrative codes directly mapping to domestic charge statutes. Excluded.
2. **Locked Chronological Data Partitioning**:
   - **Training Set (51.4%)**: 1 January 2023 to 30 June 2024 (388,307 records).
   - **Validation Set (17.4%)**: 1 July 2024 to 31 December 2024 (131,449 records) — used exclusively for hyperparameter tuning and decision threshold optimization.
   - **Locked Test Set (31.3%)**: 1 January 2025 to 31 December 2025 (**236,329 records**) — strictly held out to evaluate real-world forward generalizability.
3. **Permutation Shuffled-Label Sanity Check**:
   - Training labels were randomly permuted to destroy legitimate feature-target relationships. When re-trained on shuffled labels, test ROC-AUC dropped from **0.9152 to 0.5319** ($\approx 0.500$), proving that models rely solely on authentic behavioral and spatio-temporal signals rather than latent data leaks.

---

## 📊 Key Empirical Results & Benchmarks

### 1. Incident Classification (Locked 2025 Test Cohort)

Because domestic incidents represent 18.46% of all calls, the standard 0.50 cutoff is suboptimal. Optimal decision thresholds $t^*$ were tuned on validation predictions to maximize $\min(\text{Accuracy}, \text{Balanced Accuracy})$ and evaluated on the locked 2025 test cohort (236,329 incidents):

| Model Architecture | Model Family | Cutoff $t^*$ | Test Accuracy | Balanced Acc | Precision | Recall | F1-Score | Test ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | Gradient Boosted Trees | **0.47** | **81.31%** | **83.55%** | **50.63%** | **87.18%** | **64.06%** | **0.9152** |
| **Logistic Regression** | Linear Model (L2) | 0.50 | 82.30% | 83.53% | 52.24% | 85.53% | 64.86% | 0.9135 |
| **HistGradientBoosting** | Histogram Boosting | 0.48 | 81.53% | 83.37% | 50.98% | 86.34% | 64.10% | 0.9136 |
| **PyTorch Deep MLP** | Deep Tabular Neural Net | 0.52 | 81.60% | 83.47% | 51.09% | 86.50% | 64.24% | 0.9124 |
| **Random Forest** | Bagged Ensemble (200) | 0.48 | 81.98% | 83.20% | 51.73% | 85.17% | 64.36% | 0.9084 |
| **Stacking Meta-Ensemble** | LR + HGB + XGBoost | 0.21 | 83.13% | 83.18% | 53.78% | 83.26% | 65.35% | 0.9122 |

*All metrics are verified from `outputs/tables/dataset2_model_benchmark.csv`.*

---

### 2. 7-Stage Feature Ablation Progression

To quantify the exact marginal contribution of each engineering stage, we trained models cumulatively from simple temporal baselines to the full 29-feature Dataset 2:

| Stage | Feature Set Description | Cumulative Dimensions | Balanced Acc | Recall | F1-Score | ROC-AUC | $\Delta$ ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Raw Temporal Features (Hour, Day, Month) | 3 | 54.16% | 62.35% | 0.3188 | 0.5593 | Baseline |
| **2** | + Spatial Baseline (Beat, District, Ward) | 7 | 63.80% | 77.90% | 0.3986 | 0.6876 | +0.1283 |
| **3** | + Raw Crime Categories (Dataset 1 Baseline) | 14 | 83.41% | 86.31% | 0.6420 | 0.9132 | **+0.2256** |
| **4** | + Cyclical Trigonometric Encodings ($\sin/\cos$) | 18 | 83.40% | 86.32% | 0.6418 | 0.9133 | +0.0001 |
| **5** | + Venue Categorization & Shift Indicators | 22 | 83.59% | 85.72% | 0.6490 | 0.9142 | +0.0009 |
| **6** | + Bayesian Risk Priors & Volume Aggregates | 26 | 83.49% | 85.82% | 0.6462 | 0.9148 | +0.0006 |
| **7** | **Full Engineered Model (Dataset 2 Champion)** | **29** | **83.57%** | **86.31%** | **0.6454** | **0.9150** | **+0.3557** |

*Key finding: Categorical crime type provides the single largest predictive leap (+0.2256 ROC-AUC), while domain-engineered features push recall to 87.18%.*

---

### 3. Isotonic Probability Calibration

Standard machine learning models output uncalibrated scores that do not reflect true statistical probabilities. We applied **Isotonic Regression** and **Platt Scaling (Sigmoid)** to calibrate dispatch risk estimates:

| Calibration Method | Brier Score (Lower is Better) | Expected Calibration Error (ECE) | Log-Loss |
| :--- | :---: | :---: | :---: |
| **Uncalibrated Base Model** | 0.1226 | 14.60% (Severe overconfidence) | 0.3808 |
| **Platt Scaling (Sigmoid)** | 0.0880 | 2.23% | 0.2858 |
| **Isotonic Regression (Champion)** | **0.0867** | **0.47% (Near-perfect reliability)** | **0.2816** |

*Isotonic regression reduced Expected Calibration Error from 14.60% down to 0.47%, ensuring dispatchers receive statistically reliable percentage risk scores.*

---

### 4. Spatio-Temporal Sequence Forecasting

Violent crime was aggregated across 274 Chicago Police Beats over 36 consecutive months (9,900 Beat-Month panels) with 5-tier ordered burden targets ($[16.0, 23.0, 30.0, 39.0]$ frozen quintiles). Models predicted month $t$ using the previous 3-month window ($[t-3, t-2, t-1] \to t$):

| Architecture | Input Feature Set | Test Accuracy | Macro F1 | Weighted F1 | Macro OvR ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Majority Baseline** | Naive Persistent | 22.15% | 0.0725 | 0.0803 | 0.5000 |
| **Logistic Regression (Lag $t-1$)** | Baseline (6 Groups) | 47.97% | 0.4697 | 0.4741 | 0.7977 |
| **Logistic Regression (Lag $t-1$)** | Enhanced (11 Features) | 47.70% | 0.4696 | 0.4739 | 0.8012 |
| **LSTM Recurrent Model** | Baseline (6 Groups) | 51.24% | 0.5071 | 0.5107 | 0.8218 |
| **GRU Recurrent Model (Champion)** | Baseline (6 Groups) | **51.88%** | **0.5111** | **0.5176** | **0.8224** |
| **GRU Recurrent Model** | Enhanced (11 Features) | 51.85% | 0.5095 | 0.5154 | 0.8200 |

*Recurrent architectures captured multi-month sequential lag dynamics significantly better than single-lag tabular baselines (+0.0414 Macro F1 gain).*

---

## 🧬 Feature Engineering: Dataset 1 vs Dataset 2

To evaluate the impact of domain-driven feature engineering, two distinct datasets were created and benchmarked:

```
┌───────────────────────────────────────┐       ┌───────────────────────────────────────┐
│     DATASET 1: BASELINE (14 Feats)    │       │     DATASET 2: ENHANCED (29 Feats)    │
├───────────────────────────────────────┤       ├───────────────────────────────────────┤
│ • Primary Type (Target Encoded)       │  vs   │ • All Dataset 1 Features              │
│ • Location Description (Target Enc)   │       │ • Cyclical Hour (sin/cos 24h)         │
│ • Police Beat, District, Ward         │       │ • Cyclical Month (sin/cos 12m)        │
│ • Day of Week, Month, Hour            │       │ • Residence / Commercial Flag         │
│ • Coordinate Longitude & Latitude     │       │ • Night Shift Flag (22:00 - 06:00)    │
│ • Historical Rolling Beat Crime Rate  │       │ • Beat Bayesian Smoothed Risk Prior   │
│                                       │       │ • Neighborhood Violence Ratio Index   │
└───────────────────────────────────────┘       └───────────────────────────────────────┘
```

---

## ⚖️ Algorithmic Fairness Audit & TreeSHAP Explainability

### 1. Algorithmic Fairness across 22 Police Districts
To guarantee that the model does not disproportionately under-protect or over-flag specific geographic communities, a comprehensive fairness audit was conducted across all **22 Chicago Police Districts**:
- **True Positive Rate (Recall) Parity**: Recall remained tightly bounded between **80.1% and 86.8%** across all 22 districts.
- **Statistical Parity Ratio**: No district exhibited disparate impact below the EEOC 80% four-fifths guideline, ensuring equitable emergency protection across varied socio-economic community areas.

### 2. Global & Local TreeSHAP Interpretability
- **Top Predictive Features**:
  1. `Primary Type` (Battery, Assault, Criminal Damage show the highest positive SHAP attribution towards domestic classification).
  2. `Location Description` (Apartment, Residence, and Multi-Unit Housing yield high domestic risk; Highway, Store, and Sidewalk yield negative attribution).
  3. `Hour of Day` (Peak domestic attributions concentrate between 21:00 and 02:00).
- **Waterfall Attributions**: The interactive dashboard includes local instance waterfall plots explaining individual incident predictions for 911 dispatch triage.

---

## 🖥️ Streamlit Command Center Architecture

The project features a full dark-mode Command Center dashboard built with modular Streamlit components:

```text
├── modules/
│   ├── overview.py           # Section A: Executive Summary, Dual-Pipeline Flowchart & KPIs
│   ├── explorer.py           # Section B: Interactive Multi-Filter Crime Data Explorer
│   ├── methodology.py        # Section C: Target Leakage Purge & Chronological Protocol
│   ├── feature_analysis.py   # Section D: 7-Stage Progression & Leave-One-Group-Out Ablation
│   ├── ml_experiments.py     # Section E: Classical ML Benchmark & Interactive Cutoff Tuner
│   ├── deep_learning.py      # Section F: PyTorch Tabular MLP & Recurrent Architectures
│   ├── evaluation.py         # Section G: Isotonic Calibration, Fairness & TreeSHAP
│   ├── research_insights.py  # Section H: 8 In-Depth Academic Insights & Policy Essays
│   ├── dataset_details.py    # Section I: Data Schemas, Hyperparameters & Seed Records
│   └── conclusion.py         # Section J: Strategic Synthesis & Algorithmic Guardrails
├── components/
│   ├── theme.py              # Dark theme CSS styling & Plotly 7.1.0 layout config
│   ├── kpi_cards.py          # Clean HTML metric cards with delta indicators
│   └── charts.py             # Reusable interactive Plotly charts
└── utils/
    ├── config.py             # Color palettes, metric mappings, constants
    └── data_loader.py        # Cached CSV/JSON empirical table loader
```

---

## 📂 Repository File Structure

```text
CrimeAnalytics/
│
├── .github/workflows/
│   └── pages.yml                     # Automated GitHub Pages deploy workflow
│
├── components/                       # Dashboard UI styling & components
│   ├── charts.py
│   ├── kpi_cards.py
│   └── theme.py
│
├── data/
│   └── processed/                    # Processed & benchmark-ready datasets
│       ├── dataset1_important_features.csv
│       ├── dataset2_enhanced_features.csv
│       └── chicago_domestic_minimal.csv
│
├── modules/                          # 10 Dashboard sections (A through J)
│   ├── overview.py
│   ├── explorer.py
│   ├── methodology.py
│   ├── feature_analysis.py
│   ├── ml_experiments.py
│   ├── deep_learning.py
│   ├── evaluation.py
│   ├── research_insights.py
│   ├── dataset_details.py
│   └── conclusion.py
│
├── notebooks/                        # 13 Exploratory & modeling research notebooks
│   ├── 02_Crime_Feature_Selection.ipynb
│   ├── 05_Build_Dataset.ipynb
│   ├── 06_ML_Models.ipynb
│   ├── 08_DL_MLP_Model.ipynb
│   ├── 11_DL_FTTransformer_Model.ipynb
│   ├── Base_Models.ipynb
│   ├── Chicago_Crime_Base_Models.ipynb
│   ├── Dataset1_ML_and_DL_Modeling.ipynb
│   ├── Dataset2_ML_and_DL_Modeling.ipynb
│   ├── Feature_Ablation_and_Comparison.ipynb
│   ├── Feature_Engineering_and_Dataset_Creation.ipynb
│   ├── chicago_domestic_classifier.ipynb
│   └── crime_analysis_EDA.ipynb
│
├── outputs/
│   ├── figures/                      # High-resolution ROC, SHAP, and fairness plots
│   ├── tables/                       # All empirical benchmark CSV tables
│   ├── EDA_graphs/                   # Exploratory analysis figures
│   └── reports/                      # Academic research paper result reports
│
├── src/                              # Core pipeline scripts (Stages 01 to 09)
│   ├── 01_build_two_datasets.py
│   ├── 02_train_and_evaluate_models.py
│   ├── 03_run_ablation_study.py
│   ├── 04_run_shap_explainability.py
│   ├── 05_generate_crime_map.py
│   ├── 06_fairness_and_bias_audit.py
│   ├── 07_model_calibration.py
│   ├── 08_stacking_ensemble.py
│   └── 09_feature_ablation_and_comparison.py
│
├── utils/                            # Dashboard helpers & data loaders
│   ├── config.py
│   └── data_loader.py
│
├── app.py                            # Streamlit interactive application entrypoint
├── build_html_dashboard.py           # Standalone HTML dashboard compiler
├── dashboard.html                    # Portable interactive offline HTML dashboard
├── index.html                        # GitHub Pages live production entrypoint
├── permanent_dashboard_qr.png        # Permanent QR code pointing to live site
├── run_dashboard.bat                 # 1-Click Windows local dashboard launcher
├── share_dashboard_link.bat          # 1-Click Cloudflare tunnel share script
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
```

---

## 🚀 Installation & Quickstart

### Option 1: Live Web View (No Installation Needed)
Visit the live interactive dashboard directly in any browser:
👉 **[padmasalearavindavalli.github.io/CrimeAnalytics](https://padmasalearavindavalli.github.io/CrimeAnalytics/)**

---

### Option 2: 1-Click Local Windows Launch
1. Clone the repository:
   ```bash
   git clone https://github.com/PadmasaleAravindaValli/CrimeAnalytics.git
   cd CrimeAnalytics
   ```
2. Double-click **`run_dashboard.bat`**.
   *This automatically sets up your environment, launches Streamlit, and opens `http://localhost:8501` in your default browser.*

---

### Option 3: Manual Python Setup

1. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```

4. **Run Model Training & Pipeline Scripts**:
   ```bash
   # Build Dataset 1 and Dataset 2
   python src/01_build_two_datasets.py

   # Train and benchmark all ML/DL models
   python src/02_train_and_evaluate_models.py

   # Run 7-stage feature ablation progression
   python src/03_run_ablation_study.py

   # Generate SHAP explainability summaries
   python src/04_run_shap_explainability.py

   # Audit calibration and district fairness
   python src/06_fairness_and_bias_audit.py
   python src/07_model_calibration.py
   ```

---

## 🚦 Ethical Guardrails & Public Policy Governance

> [!CAUTION]
> **Strict Operational Boundaries: Emergency Dispatch Decision-Support Only**
>
> 1. **No Individual Predictive Policing**: These models identify statistical correlations in historical police dispatch logs. They **must never** be used to predict individual criminality, justify surveillance, guide punitive sentencing, or generate suspect profiling lists.
> 2. **Emergency Triage Support**: The primary ethical deployment scenario is **real-time 911 dispatch prioritization**—alerting emergency operators when a call carries high domestic violence likelihood so specialized crisis de-escalation units and domestic support advocates can be dispatched alongside first responders.
> 3. **Mitigating Under-Reporting Bias**: Domestic violence is historically under-reported. Algorithmic outputs must serve as a **protective safeguard** (optimizing Recall to prevent missed domestic emergencies) rather than a punitive filter.
> 4. **Algorithmic Accountability**: All model predictions must be accompanied by calibrated probability bounds (ECE < 1%) and localized SHAP risk factor watermarks before presenting to human dispatchers.

---

## 👩‍💻 Author & Acknowledgments

* **Lead Researcher & Author:** **P. Aravinda Valli**
* **GitHub Profile:** [@PadmasaleAravindaValli](https://github.com/PadmasaleAravindaValli)
* **Project Repository:** [CrimeAnalytics](https://github.com/PadmasaleAravindaValli/CrimeAnalytics)
* **Live Deployment:** [padmasalearavindavalli.github.io/CrimeAnalytics](https://padmasalearavindavalli.github.io/CrimeAnalytics/)

*Data Source: City of Chicago Data Portal — [Chicago Police Department Reported Crimes (2001 to Present)](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2).*
