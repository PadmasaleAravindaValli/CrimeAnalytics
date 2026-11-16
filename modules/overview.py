"""
Module A: Research Overview & Project Command Center.
Presents the high-level research architecture, dual-pipeline distinction, and empirical KPIs.
"""
import streamlit as st
from components.kpi_cards import render_kpi_row
from utils.config import COLORS

def render_overview():
    """Renders Section A: Research at a Glance."""
    st.markdown("""
    <div class="header-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <span class="badge-pill badge-emerald">CAPSTONE RESEARCH PLATFORM</span>
                <span class="badge-pill badge-teal" style="margin-left: 6px;">PEER-REVIEW COMPLIANT</span>
                <h1 style="margin: 8px 0 4px 0; font-size: 1.85rem; font-weight: 700; color: #f8fafc;">
                    Comparative Evaluation of Machine Learning and Deep Learning for Crime Data Classification
                </h1>
                <p style="color: #94a3b8; font-size: 0.95rem; margin: 0; max-width: 950px; line-height: 1.5;">
                    An exhaustive empirical research framework comparing classical tree ensembles, deep tabular neural architectures, 
                    recurrent sequence models, probability calibration, and algorithmic fairness across two rigorous, leak-free public safety pipelines.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Row 1
    kpis = [
        {
            "title": "Total Records Analyzed",
            "value": "765,716",
            "subtitle": "756,085 Chicago + 9,631 NCRB",
            "badge": "Dual Datasets",
            "badge_type": "cyan"
        },
        {
            "title": "Chicago Champion ROC-AUC",
            "value": "0.9152",
            "subtitle": "XGBoost (Dataset 2, 29 Features)",
            "delta": "+0.3557 vs Temporal",
            "delta_positive": True,
            "badge": "Champion",
            "badge_type": "emerald"
        },
        {
            "title": "Domestic Detection Recall",
            "value": "87.18%",
            "subtitle": "Threshold 0.47 (Balanced Acc 83.55%)",
            "delta": "Class Imbalance Solved",
            "delta_positive": True,
            "badge": "Safety Critical",
            "badge_type": "teal"
        },
        {
            "title": "Calibration Error (ECE)",
            "value": "0.47%",
            "subtitle": "Reduced from 14.60% (Isotonic)",
            "delta": "-14.13% Brier: 0.0867",
            "delta_positive": True,
            "badge": "Trustworthy ML",
            "badge_type": "amber"
        },
        {
            "title": "NCRB 5-Seed Macro F1",
            "value": "0.7097",
            "subtitle": "LSTM Enhanced (3-Yr Sequences)",
            "delta": "Leak-Free Temporal",
            "delta_positive": True,
            "badge": "Multi-Seed DL",
            "badge_type": "emerald"
        }
    ]
    render_kpi_row(kpis)
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Dual Pipeline Cards
    st.markdown("### 🏛️ Dual Research Pipeline Architecture")
    st.markdown("""
    The research investigates two distinct crime classification paradigms to evaluate whether machine learning and deep learning 
    generalize across both micro-level incident triage and macro-level administrative sequence forecasting:
    """)
    
    p1, p2 = st.columns(2)
    
    with p1:
        st.markdown(f"""
        <div class="panel-card" style="border-top: 3px solid {COLORS['emerald']};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h4 style="margin: 0; color: #f8fafc;">Pipeline 1: Chicago Incident Classification</h4>
                <span class="badge-pill badge-emerald">Micro-Incident Triage</span>
            </div>
            <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
                <b>Dataset:</b> Chicago Police Department Incident Records (2023–2025).<br>
                <b>Observations:</b> 756,085 validated incidents across 22 police districts.<br>
                <b>Prediction Task:</b> Binary classification of domestic violence altercations (<code>Domestic = 1/0</code>).<br>
                <b>Class Prevalence:</b> 18.46% domestic (significant minority class imbalance).<br>
                <b>Temporal Split:</b> Train (388k, 2023–June 2024), Val (131k, July–Dec 2024), Test (236k, 2025).<br>
                <b>Key Challenge:</b> Severe operational target leakage in police text records (143k rows contain "DOMESTIC" in crime description) + post-arrest features.
            </p>
            <div style="background-color: {COLORS['surface_light']}; padding: 10px; border-radius: 6px; font-size: 0.8rem; color: #cbd5e1;">
                <b>Models Evaluated:</b> Logistic Regression, Random Forest, HistGradientBoosting, XGBoost (Champion), Deep Tabular MLP, Stacking Meta-Ensemble.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with p2:
        st.markdown(f"""
        <div class="panel-card" style="border-top: 3px solid {COLORS['teal']};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h4 style="margin: 0; color: #f8fafc;">Pipeline 2: Indian NCRB Burden Forecasting</h4>
                <span class="badge-pill badge-teal">Macro-District Sequence</span>
            </div>
            <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
                <b>Dataset:</b> National Crime Records Bureau (NCRB) District-Year Statistics (2001–2014).<br>
                <b>Observations:</b> 9,856 district-years consolidated across 35 States / UTs.<br>
                <b>Prediction Task:</b> 5-tier ordered violent crime burden category (<code>TARGET_VIOLENT_LEVEL</code>).<br>
                <b>Frozen Thresholds:</b> Fitted on 2001–2010 training quintiles (107, 237, 362, 563).<br>
                <b>Chronological Split:</b> Train (2001–2010), Val (2011–2012), Locked Test (2013–2014, 671 shared sequences).<br>
                <b>Sequence Design:</b> 3-year sliding lag window (\(t-3, t-2, t-1 \\to t\)) across consecutive district records.
            </p>
            <div style="background-color: {COLORS['surface_light']}; padding: 10px; border-radius: 6px; font-size: 0.8rem; color: #cbd5e1;">
                <b>Models Evaluated:</b> Multi-seed GRU, LSTM, Logistic Regression, Majority Baseline + target leakage audit of legacy MLP (90.76%).
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Visual Workflow Container
    st.markdown("### 🔄 End-to-End Experimental Methodology Flowchart")
    st.markdown("""
    ```mermaid
    graph LR
        subgraph Pipeline 1 [Chicago Incident Classification]
            A1["Raw CPD Records<br>2023-2025 (756k)"] --> B1["Leakage Audit<br>Drop Description & Arrest"]
            B1 --> C1["Feature Engineering<br>D1 (14) vs D2 (29)"]
            C1 --> D1["Chronological Split<br>Train: 2023-24H1 | Test: 2025"]
            D1 --> E1["Model Benchmark<br>XGBoost, DL MLP, Stacking"]
            E1 --> F1["Calibration & Fairness<br>Isotonic ECE: 0.47% | 22 Dists"]
        end

        subgraph Pipeline 2 [Indian NCRB Sequence Forecasting]
            A2["Raw NCRB District Records<br>2001-2014 (9.8k)"] --> B2["6 IPC Crime Groups<br>+ Target Quintiles"]
            B2 --> C2["5 Dynamic Features<br>Diversity, YoY Change, etc."]
            C2 --> D2["Temporal 3-Yr Sequences<br>Locked Test 2013-14 (671 seqs)"]
            D2 --> E2["Recurrent Sequence Models<br>GRU vs LSTM (5 Seeds)"]
            E2 --> F2["Multi-Seed Ablation<br>LSTM Gain: +0.0125 Macro F1"]
        end
    ```
    """)
    
    # Research Principles Expander
    with st.expander("🔬 Core Methodological Principles & Compliance Standards", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div style="border-left: 2px solid {COLORS['emerald']}; padding-left: 12px;">
                <b style="color: #f8fafc;">1. Zero Target Leakage</b>
                <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    Both pipelines enforce strict temporal contracts. In Chicago, all post-incident investigation fields, 
                    arrest flags, and descriptive free-text containing "DOMESTIC" were audited and permanently purged.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="border-left: 2px solid {COLORS['teal']}; padding-left: 12px;">
                <b style="color: #f8fafc;">2. Future-Time Evaluation</b>
                <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    No random k-fold or random stratified splits were permitted for primary model selection. Models were trained on historical records 
                    and strictly evaluated on subsequent, unseen future time periods.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div style="border-left: 2px solid {COLORS['cyan']}; padding-left: 12px;">
                <b style="color: #f8fafc;">3. Multi-Metric Rigor</b>
                <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    Because raw accuracy hides high false negative rates in safety-critical domains, all conclusions rest on 
                    Balanced Accuracy, Recall, Macro F1, Brier Score, ECE, and Disparate Impact ratios.
                </p>
            </div>
            """, unsafe_allow_html=True)
