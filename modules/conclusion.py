"""
Module J: Academic Conclusion, Ethical Guardrails & Future Scope.
Comprehensive synthesis of project achievements, policy implications, and future research directions.
"""
import streamlit as st
from utils.config import COLORS

def render_conclusion():
    """Renders Section J: Conclusion and Future Scope."""
    st.markdown("""
    <div class="header-banner">
        <span class="badge-pill badge-emerald">RESEARCH SYNTHESIS</span>
        <h2 style="margin: 6px 0 2px 0; color: #f8fafc;">Academic Conclusion & Strategic Roadmap</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
            Summary of empirical contributions, operational implications for public safety dispatch, 
            ethical boundaries, and future research opportunities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Key Pillars of Conclusion
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="panel-card" style="border-top: 3px solid {COLORS['emerald']}; height: 100%;">
            <h4 style="margin-top: 0; color: #f8fafc;">1. Methodological & Empirical Achievements</h4>
            <ul style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
                <li><b>Leakage-Free Benchmark:</b> Successfully exposed and purged catastrophic target leakage across both police text fields (Chicago) and overlapping statutory aggregates (NCRB), replacing inflated vanity metrics with genuine future-year forecasting.</li>
                <li><b>Champion Discrimination:</b> XGBoost achieved <b>0.9152 ROC-AUC</b> and <b>87.18% Recall</b> on Chicago incident dispatch; LSTM achieved <b>0.7097 Macro F1</b> across 5 random seeds on NCRB sequence forecasting.</li>
                <li><b>Trustworthy Probabilities:</b> Reduced Expected Calibration Error from <b>14.60% to 0.47%</b> using Isotonic Regression, transforming raw algorithmic margins into actionable probabilities.</li>
                <li><b>Demographic Parity:</b> Audited 22 police districts, proving stable True Positive Rates (80%–87%) across socio-demographically diverse neighborhoods.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="panel-card" style="border-top: 3px solid {COLORS['rose']}; height: 100%;">
            <h4 style="margin-top: 0; color: #f8fafc;">2. Ethical Guardrails & Governance Policy</h4>
            <ul style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
                <li><b>Decision Support, Never Autonomous Action:</b> These models are intended strictly as decision-support tools for emergency dispatchers to prompt safety protocols (e.g., dispatching domestic violence specialist units).</li>
                <li><b>Strict Rejection of Predictive Policing:</b> This framework explicitly rejects proactive hotspot targeting, individual recidivism scoring, and automated surveillance.</li>
                <li><b>Awareness of the 'Dark Figure of Crime':</b> Police-recorded data fundamentally reflects reporting behavior and administrative discretion rather than true underlying crime incidence. Unreported domestic abuse remains a known structural limitation.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🚀 Future Research Roadmap")
    
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="panel-card" style="border-left: 3px solid {COLORS['cyan']};">
            <h5 style="margin-top: 0; color: #f8fafc;">Spatio-Temporal GNNs</h5>
            <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                Constructing explicit road-network and police beat adjacency graphs to deploy Spatio-Temporal Graph Neural Networks (ST-GNNs) 
                capable of modeling spatial diffusion between contiguous urban sectors.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with r2:
        st.markdown(f"""
        <div class="panel-card" style="border-left: 3px solid {COLORS['teal']};">
            <h5 style="margin-top: 0; color: #f8fafc;">Multimodal 911 Audio</h5>
            <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                Integrating acoustic speech biomarkers, caller stress tone, and automated dispatch transcription using pre-trained audio transformers 
                while preserving strict privacy and evidentiary security.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with r3:
        st.markdown(f"""
        <div class="panel-card" style="border-left: 3px solid {COLORS['purple']};">
            <h5 style="margin-top: 0; color: #f8fafc;">Continuous Online Calibration</h5>
            <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                Implementing streaming probability recalibration mechanisms to dynamically adapt to seasonal shifts, policy reforms, 
                and reporting rate fluctuations without requiring full batch retraining.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; border-top: 1px solid {COLORS['border']}; color: #64748b; font-size: 0.8rem;">
        <b>Comparative Evaluation of Machine Learning and Deep Learning for Crime Data Classification</b><br>
        Developed as an Interactive Research Analytics Platform & Capstone Technical Demonstration &bull; 2026<br>
        Designed with Streamlit, Plotly, Scikit-Learn, PyTorch & XGBoost.
    </div>
    """, unsafe_allow_html=True)
