"""
Module G: Model Evaluation, Probability Calibration & Algorithmic Fairness.
Details calibration curves (ECE reduction), 22-district fairness audit, and SHAP explainability.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_chicago_tables
from utils.config import COLORS
from components.theme import get_plotly_layout
from components.charts import (
    create_calibration_chart,
    create_fairness_parity_chart,
    create_shap_bar_chart,
)

def render_evaluation():
    """Renders Section G: Model Evaluation, Calibration & Explainability."""
    st.markdown("""
    <div class="header-banner">
        <span class="badge-pill badge-amber">TRUSTWORTHY MACHINE LEARNING</span>
        <h2 style="margin: 6px 0 2px 0; color: #f8fafc;">Evaluation, Calibration & Algorithmic Fairness</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
            Beyond aggregate accuracy: Verifying probability reliability, evaluating disparate impact across 22 police districts, 
            and interpreting decision logic using SHapley Additive exPlanations (SHAP).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    eval_tab1, eval_tab2, eval_tab3 = st.tabs([
        "🎯 Probability Calibration (ECE)",
        "⚖️ District Algorithmic Fairness Audit",
        "🔍 SHAP Explainability & Feature Drivers"
    ])
    
    chicago_tables = load_chicago_tables()
    df_cal = chicago_tables.get("calibration", pd.DataFrame())
    df_fair = chicago_tables.get("fairness", pd.DataFrame())
    df_shap = chicago_tables.get("shap", pd.DataFrame())
    
    # =========================================================================
    # TAB 1: PROBABILITY CALIBRATION
    # =========================================================================
    with eval_tab1:
        st.markdown("### 📉 Probability Calibration & Expected Calibration Error (ECE)")
        st.markdown("""
        In real-time public safety dispatch, the model's output is not merely a binary flag; dispatchers and supervisors 
        require trustworthy probabilities to prioritize resource triage. Gradient boosted trees produce uncalibrated, 
        over-confident margins that distort risk assessment.
        """)
        
        c_col1, c_col2 = st.columns([1.3, 1])
        with c_col1:
            st.plotly_chart(create_calibration_chart(), use_container_width=True)
            
        with c_col2:
            st.markdown("#### 📋 Empirical Calibration Ledger")
            if not df_cal.empty:
                st.dataframe(df_cal, use_container_width=True, hide_index=True)
            st.markdown(f"""
            <div class="panel-card" style="margin-top: 15px;">
                <b style="color: {COLORS['emerald']}; font-size: 0.95rem;">Key Takeaway:</b>
                <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 6px; line-height: 1.5;">
                    Isotonic Regression dropped the Expected Calibration Error from <b>14.60% down to 0.47%</b>, 
                    representing an enormous <b>96.8% reduction in miscalibration</b>. Brier score decreased from 0.1226 to 0.0867. 
                    A predicted risk score of 0.70 now truly represents a 70% empirical probability of domestic violence.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: DISTRICT ALGORITHMIC FAIRNESS AUDIT
    # =========================================================================
    with eval_tab2:
        st.markdown("### ⚖️ Algorithmic Fairness Audit Across 22 Police Districts")
        st.markdown("""
        To verify whether the champion XGBoost classifier exhibits disparate impact or unequal error rates across socio-demographically 
        diverse urban environments, we audited all 22 active Chicago Police Districts on the 2025 Test Set (236,329 incidents):
        """)
        
        if not df_fair.empty:
            st.plotly_chart(create_fairness_parity_chart(df_fair), use_container_width=True)
            
            with st.expander("📋 View Complete 22-District Fairness Ledger", expanded=False):
                st.dataframe(df_fair, use_container_width=True, hide_index=True)
                
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.markdown(f"""
                <div class="panel-card">
                    <h5 style="margin-top: 0; color: #f8fafc;">1. Equal Opportunity (TPR Parity)</h5>
                    <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                        Across 20 of the 22 districts, the True Positive Rate (Recall) remains strictly bounded between 
                        <b>80.1% and 86.8%</b>. The model detects domestic violence incidents with virtually identical sensitivity 
                        regardless of whether the altercation occurs in the South Side, West Side, or North Side.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with f_col2:
                st.markdown(f"""
                <div class="panel-card">
                    <h5 style="margin-top: 0; color: #f8fafc;">2. Disparate Impact & FPR Analysis</h5>
                    <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                        The Disparate Impact Ratio ranges from 0.375 (District 1: Central Business Loop, low ambient domestic crime) 
                        to 1.547 (District 15: Austin, high baseline volume). False Positive Rates are slightly elevated in high-volume districts, 
                        warranting calibrated threshold offsets in operational deployment.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Fairness audit data not available.")

    # =========================================================================
    # TAB 3: SHAP EXPLAINABILITY
    # =========================================================================
    with eval_tab3:
        st.markdown("### 🔍 SHAP (SHapley Additive exPlanations) Global Importance")
        st.markdown("""
        TreeSHAP was computed on a representative sample of the 2025 Test Set to attribute marginal log-odds contributions 
        to each engineered feature without post-hoc heuristic bias:
        """)
        
        if not df_shap.empty:
            s_c1, s_c2 = st.columns([1.3, 1])
            with s_c1:
                st.plotly_chart(create_shap_bar_chart(df_shap), use_container_width=True)
            with s_c2:
                st.markdown("#### 💡 Interpretability Findings")
                st.markdown(f"""
                <div class="panel-card">
                    <ol style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6; padding-left: 18px;">
                        <li><b>Primary Bayesian Risk Prior (1.785 SHAP):</b> The historical domestic likelihood conditional on statutory offense type is by far the single strongest driver. Offenses like Battery and Assault carry massive positive attribution.</li>
                        <li><b>Location Domestic Risk (1.149 SHAP):</b> Private residences, apartments, and domestic premises strongly increase domestic classification probability.</li>
                        <li><b>Beat Domestic Risk (0.381 SHAP):</b> Micro-geographic neighborhood baseline domestic altercations add meaningful spatial context.</li>
                        <li><b>Trigonometric Time (Hour_Cos, Hour_Sin):</b> Nighttime and late-evening hours exert positive pushes towards domestic altercations, whereas business hours depress the score.</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("SHAP data not available.")
