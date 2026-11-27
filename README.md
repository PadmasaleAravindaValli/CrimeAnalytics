# CrimeAnalytics — Computational Analysis of Crime and Social Systems

[![Python 3.11](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-185a9d.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Dashboard-brightgreen.svg)](https://padmasalearavindavalli.github.io/CrimeAnalytics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computational crime analytics research platform and empirical comparative study evaluating incident-level domestic crime classification (Chicago Police Department) and district-level spatio-temporal violent crime sequence forecasting (National Crime Records Bureau, India).

---

## 🌐 Live Interactive Research Dashboard (v2.0 Minimal & Clean)

The complete empirical research findings, interactive operating threshold simulator, 7-stage cumulative feature progression charts, probability calibration curves, and 22-district algorithmic fairness audits are deployed live:

* **Permanent Web URL:** **[padmasalearavindavalli.github.io/CrimeAnalytics](https://padmasalearavindavalli.github.io/CrimeAnalytics/)** *(Zero-code, accessible worldwide on mobile, tablet, and PC)*
* **Permanent Static QR Code:**
  
  <p align="center">
    <img src="permanent_dashboard_qr.png" alt="Permanent Dashboard QR Code" width="220" />
    <br>
    <em>Scan with any smartphone camera to launch the live research platform</em>
  </p>

* **Dashboard Redesign Highlights (v2.0 Minimal):**
  * **5 Streamlined Tabs:** Executive Overview & Architecture, Chicago CPD Classification, Indian NCRB Forecasting, Fairness & Calibration, and Data & Paper Downloads.
  * **Interactive Threshold Simulator:** Dynamic live slider ($\tau \in [0.10, 0.90]$) adjusting Accuracy, Balanced Accuracy, Precision, Domestic Recall, and F1 in real time.
  * **1-Click Research Downloads:** Direct browser downloads for `CrimeAnalytics_paper.docx`, `research_paper_lncs.tex`, and 4 benchmark CSV datasets.
  * **Original Layout Backup:** Safely preserved in `index_original_backup.html` and `dashboard_original_backup.html`.

---

## 📑 Dual-Pipeline Research Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        DUAL-PIPELINE COMPREHENSIVE CRIME ANALYTICS ARCHITECTURE                        │
├───────────────────────────────────────────────────┬────────────────────────────────────────────────────┤
│ PIPELINE A: CHICAGO CPD INCIDENT CLASSIFICATION   │ PIPELINE B: INDIAN NCRB SEQUENTIAL FORECASTING     │
│ (756,085 Clean Records | Forward 2025 Test Cohort)│ (5,842 Clean Sequences | Forward 2013-14 Test Set) │
│                                                   │                                                    │
│  [Raw CPD Incidents]                              │  [District Crime Panel]                            │
│         │ (757,898 raw cases)                     │         │ (9,631 target rows)                      │
│         ▼                                         │         ▼                                          │
│  [Target Leakage Purge]                           │  [History Data Cleaning]                           │
│  Drop Description (143k tokens), IUCR, Arrest     │  Drop 2,469 incomplete & 1,320 non-finite rows     │
│         ▼                                         │         ▼                                          │
│  [15 Domain Features Engineered]                  │  [11 Sequential Trend Features]                    │
│  Cyclic (6), Venue (4), Priors (4), Inter (1)     │  Diversity, YoY growth, volume share, disparity    │
│         ▼                                         │         ▼                                          │
│  [Champion XGBoost: AUC 0.9152, Recall 87.18%]    │  [5-Seed Recurrent Models: LSTM 0.7097, GRU 0.7083]│
│         ▼                                         │         ▼                                          │
│  [Isotonic Calibration: ECE 14.60% → 0.47%]       │  [Paired t-test: t = 0.345, p = 0.747 (Equiv.)]   │
│         ▼                                         │         ▼                                          │
│  [911 Dispatch Triage & 22-District Audit]        │  [5-Tier Crime Burden Strategic Resource Plan]     │
├───────────────────────────────────────────────────┴────────────────────────────────────────────────────┤
│  ⚠ ETHICAL GUARDRAILS: Strictly designed for emergency dispatch triage and social resource allocation. │
│  Punitive predictive policing, individual profiling, and automated punitive enforcement are rejected.  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Empirical Contributions & Ground Truth

### 1. Reconciled Dataset Partitioning (Purged Data Leakage)
* **Raw Records:** 757,898 incidents (2021–2023 Train: 389,196; 2024 Val: 131,775; 2025 Test: 236,927).
* **Duplicate Elimination:** **1,813 co-occurring duplicate records** sharing identical coordinates, timestamps, beats, and primary types were purged.
* **Final Clean Dataset Total:** **756,085 incidents** (Train 2021–2023: 388,307; Val 2024: 131,449; Locked 2025 Test: 236,329).
* **Target Leakage Purging:** Removed post-investigation fields (`Description` containing 143k "DOMESTIC" text tokens, `IUCR`, `FBI Code`, `Arrest`) that artificially inflate naive models to >99%.

### 2. Feature Configurations (Exact Counts)
* **Dataset 1 (Core Baseline):** **11 baseline model inputs** + 3 bookkeeping columns (`Date`, `split`, `Domestic`) = **14 total schema columns**.
* **Dataset 2 (Domain Engineered):** Adds **15 domain features** across 4 groups:
  * *Cyclical Time (6):* `Hour_Sin`, `Hour_Cos`, `Month_Sin`, `Month_Cos`, `DayOfWeek_Sin`, `DayOfWeek_Cos`
  * *Venue & Shift Flags (4):* `Is_Residential`, `Is_Public_Space`, `Is_Night`, `Time_Period`
  * *Offence-Venue Interaction (1):* `Type_Location_Interaction`
  * *Bayesian Smoothed Risk Priors (4):* `Primary_Type_Domestic_Risk`, `Location_Domestic_Risk`, `Beat_Domestic_Risk` ($m=50$), and `Beat_Historical_Incident_Volume`
* **Total Dimensions:** **26 model inputs** ($11 + 15 = 26$) and **29 total schema columns** ($14 + 15 = 29$).

### 3. Chicago Classification Benchmark (2025 Test Cohort, $n = 236,329$)

| Model Architecture | Feature Variant | Tuned $\tau^*$ | Accuracy | Balanced Acc | Precision | Domestic Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **Dataset 2 (Enhanced 26)** | **0.47** | **81.31%** | **83.55%** | **50.63%** | **87.18%** | **64.06%** | **0.9152** |
| XGBoost | Dataset 1 (Baseline 11) | 0.50 | 82.08% | 83.49% | 51.87% | 85.79% | 64.65% | 0.9137 |
| PyTorch Deep Tabular MLP | Dataset 2 (Enhanced 26) | 0.52 | 81.60% | 83.47% | 51.09% | 86.50% | 64.24% | 0.9124 |
| HistGradientBoosting | Dataset 2 (Enhanced 26) | 0.48 | 81.53% | 83.37% | 50.98% | 86.34% | 64.10% | 0.9136 |
| Logistic Regression | Dataset 2 (Enhanced 26) | 0.50 | 82.30% | 83.53% | 52.24% | 85.53% | 64.86% | 0.9135 |
| Random Forest | Dataset 2 (Enhanced 26) | 0.48 | 81.98% | 83.20% | 51.73% | 85.17% | 64.36% | 0.9084 |

### 4. 7-Stage Feature Progression & LOGO Ablation
* **Stage Progression:** S1 Temporal ($0.5593$) $\to$ S2 Spatial ($0.6876$) $\to$ **S3 Crime Type ($0.9132$, $+0.2256$ jump)** $\to$ S4 Cyclical ($0.9134$) $\to$ S5 Venue ($0.9138$) $\to$ S6 Interaction ($0.9142$) $\to$ S7 Bayesian Priors ($0.9150$).
* **LOGO Ablation Penalties:** Dropping Bayesian Priors causes largest loss ($-0.0009$), followed by Spatial Features ($-0.0007$) and Interaction ($-0.0002$); Cyclical and Venue/Shift groups show no measurable loss ($\approx 0$).

### 5. Probability Calibration
* **Uncalibrated Model:** $\text{ECE} = 14.60\%$ (inflated due to `scale_pos_weight ≈ 4.43`).
* **Platt Scaling:** $\text{ECE} = 2.23\%$, Brier Score $= 0.0880$.
* **Isotonic Regression (Champion):** $\text{ECE} = \mathbf{0.47\%}$, Brier Score $= \mathbf{0.0867}$.
* *Validation Protocol:* Calibrators were fitted strictly on the 2024 validation cohort; the 2025 test set was completely untouched during fitting.

### 6. Algorithmic Fairness Audit Across 22 Police Districts
* **Disparate Impact (DI):** Relative to reference District 24, all 20 residential districts stay within the **EEOC 0.80–1.25 parity band** ($0.85$ to $1.21$).
* **Outliers:** Only commercial/downtown retail beats (District 1: Central Loop at $0.38$, District 18: Near North at $0.42$) drop below $0.80$ due to low domestic base rates.
* **Equalized Odds Transparency:** True Positive Rate ($68.1\%$–$87.9\%$) and False Positive Rate ($7.1\%$–$28.6\%$) with 95% Wilson confidence intervals confirm equalized odds is not achieved across heterogeneous urban beats.

### 7. Indian NCRB Violent Crime Sequential Forecasting
* **Clean Sequences:** 5,842 clean 3-year sequences (2004–2010 Train: 3,847; 2011–2012 Val: 1,324; 2013–2014 Test: 671).
* **5-Seed Multi-Run Evaluation:**
  * **Enhanced GRU:** $0.7083 \pm 0.0076$ Macro F1 | $0.7228 \pm 0.0064$ Accuracy
  * **Enhanced LSTM (Champion):** $\mathbf{0.7097 \pm 0.0066}$ Macro F1 | $\mathbf{0.7273 \pm 0.0061}$ Accuracy
* **Statistical Significance:** Paired t-test yields $t = 0.345, \mathbf{p = 0.747} > 0.05$ (difference between LSTM and GRU is not statistically significant; error bars overlap).

---

## 📊 Publication-Grade 300 DPI Figures (`figures/`)

All 7 publication-grade figures are embedded in the research manuscripts and saved as 300 DPI PNGs in `figures/`:

| Figure | Filename | Operational Focus |
| :--- | :--- | :--- |
| **Fig. 1** | `fig1_overall_workflow.png` | Dual-pipeline architecture with ethical guardrails banner |
| **Fig. 2** | `fig2_chicago_model_comparison.png` | Architecture discrimination gains & calibrated ROC curves at $\tau^* = 0.47$ |
| **Fig. 3** | `fig3_feature_progression_and_logo.png` | 7-Stage progression ($+0.2256$ jump) & LOGO ablation ($\approx 0$ annotated) |
| **Fig. 4** | `fig4_probability_calibration.png` | Reliability diagram: uncalibrated ($14.60\%$) vs. isotonic ($0.47\%$ ECE) |
| **Fig. 5** | `fig5_ncrb_recurrent_comparison.png` | 5-Seed LSTM vs. GRU comparison with error bars ($p = 0.747$) & feature contributions |
| **Fig. 6** | `fig6_confusion_matrices.png` | Normalized Chicago CPD ($n=236,329$) & NCRB 5-tier forecasting ($n=671$) |
| **Fig. 7** | `fig7_fairness_audit_districts.png` | 22-District disparate impact bar chart & TPR/FPR with 95% Wilson CIs |

---

## 📁 Repository Structure

```
CrimeAnalytics/
├── CrimeAnalytics_paper.docx          # Complete Word manuscript with 9 tables & 7 embedded figures
├── REVISED_RESEARCH_PAPER.md          # Comprehensive Markdown paper adhering to reviewer structure
├── research_paper_lncs.tex            # Springer LNCS LaTeX source code (Overleaf-ready)
├── index.html                         # Live minimal & clean GitHub Pages dashboard (v2.0)
├── dashboard.html                     # Offline standalone HTML dashboard
├── index_original_backup.html         # Preserved backup copy of previous dashboard layout
├── dashboard_original_backup.html     # Preserved backup copy of previous dashboard layout
├── permanent_dashboard_qr.png         # Direct static QR code linking to GitHub Pages
├── make_all_figures.py                # Python script generating all 7 300-DPI figures
├── insert_figures_to_docx.py          # Automated script embedding figures into Word docx
├── app.py                             # Streamlit interactive application
├── run_dashboard.bat                  # 1-click Windows launcher for local Streamlit dashboard
├── figures/                           # High-resolution 300 DPI publication figures
│   ├── fig1_overall_workflow.png
│   ├── fig2_chicago_model_comparison.png
│   ├── fig3_feature_progression_and_logo.png
│   ├── fig4_probability_calibration.png
│   ├── fig5_ncrb_recurrent_comparison.png
│   ├── fig6_confusion_matrices.png
│   └── fig7_fairness_audit_districts.png
├── components/                        # Streamlit visual components and theme configs
├── modules/                           # Model evaluation and prediction modules
├── utils/                             # Data loading and preprocessing utilities
├── .github/workflows/pages.yml        # Automated GitHub Pages CI/CD deployment
└── README.md                          # Repository documentation
```

---

## 🚀 Quick Start Guide

### 1. View the Live Web Dashboard
Visit **[padmasalearavindavalli.github.io/CrimeAnalytics](https://padmasalearavindavalli.github.io/CrimeAnalytics/)** in any browser.

### 2. View Offline
Double-click `index.html` or `dashboard.html` to run the fully responsive dashboard locally without an internet connection.

### 3. Run Streamlit Python App
```powershell
# Clone the repository
git clone https://github.com/PadmasaleAravindaValli/CrimeAnalytics.git
cd CrimeAnalytics

# Install requirements
pip install -r requirements.txt

# Launch Streamlit dashboard
streamlit run app.py
```

---

## 👥 Authors & Academic Affiliation
* **Padmasale Aravinda Valli** — Department of CSE (Data Science), Dayananda Sagar University, Bengaluru, India
* **Varshini B** — Department of CSE (Data Science), Dayananda Sagar University
* **Vaishnavi TP** — Department of CSE (Data Science), Dayananda Sagar University
* **Dr. Manasvi Bakshi** — Assistant Professor, Department of CSE (Data Science), Dayananda Sagar University
