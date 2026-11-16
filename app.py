"""
Main Streamlit Application: Interactive Crime Analytics & ML Research Dashboard.
Run: streamlit run app.py
"""
import streamlit as st
import sys
from pathlib import Path

# Ensure local directories are in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Streamlit Page Config
st.set_page_config(
    page_title="Crime Analytics & ML Research Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Theme and CSS
from components.theme import apply_custom_theme
from utils.config import COLORS
apply_custom_theme()

# Import Modules
from modules.overview import render_overview
from modules.explorer import render_explorer
from modules.methodology import render_methodology
from modules.feature_analysis import render_feature_analysis
from modules.ml_experiments import render_ml_experiments
from modules.deep_learning import render_deep_learning
from modules.evaluation import render_evaluation
from modules.research_insights import render_research_insights
from modules.dataset_details import render_dataset_details
from modules.conclusion import render_conclusion

# Sidebar Architecture
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #232f48; margin-bottom: 15px;">
        <span class="badge-pill badge-emerald" style="font-size: 0.7rem;">COMMAND CENTER</span>
        <h3 style="margin: 6px 0 2px 0; font-size: 1.15rem; color: #f8fafc; font-weight: 700;">
            CRIME ML RESEARCH
        </h3>
        <span style="font-size: 0.75rem; color: #94a3b8;">
            Comparative Evaluation of ML & DL
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 600; margin-bottom: 6px;'>Research Navigation</p>", unsafe_allow_html=True)
    
    nav_options = [
        "1. 🏛️ Overview & Research at a Glance",
        "2. 🔍 Crime Data Explorer",
        "3. ⚙️ Data Processing & Methodology",
        "4. 🔬 Feature Engineering & Ablation",
        "5. 🧪 Machine Learning Experiment Lab",
        "6. 🧠 Deep Learning Experiments",
        "7. 📊 Evaluation, Calibration & Fairness",
        "8. 💡 Research Findings & Insights",
        "9. 📋 Dataset & Experiment Details",
        "10. 🎯 Academic Conclusion & Roadmap"
    ]
    
    selected_page = st.radio(
        "Navigation Menu",
        nav_options,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Sidebar Live Empirical Badge
    st.markdown(f"""
    <div style="background-color: #151c2c; border: 1px solid #232f48; border-radius: 6px; padding: 12px; margin-top: 10px;">
        <span style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Empirical Highlights</span>
        <div style="margin-top: 6px; font-size: 0.8rem; color: #cbd5e1; line-height: 1.6;">
            &bull; Chicago D2 ROC-AUC: <b style="color: {COLORS['emerald']};">0.9152</b><br>
            &bull; Domestic Recall: <b style="color: {COLORS['teal']};">87.18%</b><br>
            &bull; Isotonic ECE: <b style="color: {COLORS['amber']};">0.47%</b><br>
            &bull; NCRB 5-Seed F1: <b style="color: {COLORS['cyan']};">0.7097</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: 25px; padding-top: 12px; border-top: 1px solid #232f48; font-size: 0.75rem; color: #64748b;">
        <span>Status: <b>Peer-Review Ready</b></span><br>
        <span>Environment: <b>Python 3.11 / Streamlit</b></span>
    </div>
    """, unsafe_allow_html=True)

# Main Routing Logic
if selected_page.startswith("1."):
    render_overview()
elif selected_page.startswith("2."):
    render_explorer()
elif selected_page.startswith("3."):
    render_methodology()
elif selected_page.startswith("4."):
    render_feature_analysis()
elif selected_page.startswith("5."):
    render_ml_experiments()
elif selected_page.startswith("6."):
    render_deep_learning()
elif selected_page.startswith("7."):
    render_evaluation()
elif selected_page.startswith("8."):
    render_research_insights()
elif selected_page.startswith("9."):
    render_dataset_details()
elif selected_page.startswith("10."):
    render_conclusion()
