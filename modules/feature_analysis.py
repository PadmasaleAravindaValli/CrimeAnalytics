"""
Module D: Feature Engineering Analysis & Ablation Experiments.
Analyzes 7-stage progression, Mutual Information rankings, LOGO impact, and 5-seed NCRB dynamic features.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_chicago_tables, load_ncrb_tables
from utils.config import COLORS, COLORWAY
from components.theme import get_plotly_layout
from components.charts import (
    create_feature_progression_chart,
    create_logo_ablation_chart,
    create_model_comparison_bar,
)

def render_feature_analysis():
    """Renders Section D: Feature Engineering Analysis."""
    st.markdown("""
    <div class="header-banner">
        <span class="badge-pill badge-emerald">EMPIRICAL FEATURE ABLATION</span>
        <h2 style="margin: 6px 0 2px 0; color: #f8fafc;">Feature Engineering & Ablation Laboratory</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
            Rigorous ablation studies isolating the exact performance contributions of temporal, spatial, 
            Bayesian risk prior, and dynamic sequential crime indicators on held-out test sets.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    fa_tab1, fa_tab2 = st.tabs(["🏙️ Chicago Feature Progression & Ablation", "🗺️ NCRB Dynamic Features & 5-Seed Ablation"])
    
    # =========================================================================
    # TAB 1: CHICAGO FEATURE ANALYSIS
    # =========================================================================
    with fa_tab1:
        chicago_tables = load_chicago_tables()
        df_prog = chicago_tables.get("ablation_progression", pd.DataFrame())
        df_ranking = chicago_tables.get("ranking", pd.DataFrame())
        df_logo = chicago_tables.get("leave_one_group_out", pd.DataFrame())
        df_d1_d2 = chicago_tables.get("d1_vs_d2", pd.DataFrame())
        
        # 1. 7-Stage Progression Chart
        st.markdown("### 📈 7-Stage Cumulative Feature Progression")
        st.markdown("""
        How model discrimination evolves as each feature family is incrementally integrated into XGBoost on the locked 2025 Test Set:
        """)
        
        if not df_prog.empty:
            st.plotly_chart(create_feature_progression_chart(df_prog), use_container_width=True)
            with st.expander("📊 View 7-Stage Progression Data Table", expanded=False):
                st.dataframe(df_prog, use_container_width=True, hide_index=True)
        else:
            st.info("Feature progression data not available.")
            
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        # 2. Ranking: Mutual Information vs Correlation
        st.markdown("### 🔬 Information Theoretic vs Linear Feature Rankings")
        st.markdown("""
        Comparison of linear relationship (Pearson Correlation) versus non-linear dependency (Mutual Information with Domestic target) 
        across all 19 predictive features:
        """)
        
        if not df_ranking.empty:
            r_col1, r_col2 = st.columns([1.2, 1])
            with r_col1:
                fig_scatter = px.scatter(
                    df_ranking,
                    x="Mutual_Information",
                    y="Correlation_with_Domestic",
                    color="Category",
                    text="Feature",
                    size="Abs_Correlation",
                    color_discrete_map={
                        "Bayesian Risk Prior": COLORS["emerald"],
                        "Venue / Shift Flag": COLORS["teal"],
                        "Spatial / Volume": COLORS["cyan"],
                        "Cyclical Temporal": COLORS["purple"],
                        "Raw Temporal": COLORS["amber"]
                    },
                    title="<b>Mutual Information vs Target Correlation</b>"
                )
                fig_scatter.update_traces(textposition="top center", marker=dict(sizemin=6))
                fig_scatter.update_layout(**get_plotly_layout(height=420))
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with r_col2:
                st.dataframe(
                    df_ranking[["Feature", "Mutual_Information", "Correlation_with_Domestic", "Category"]],
                    use_container_width=True,
                    hide_index=True,
                    height=420
                )
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        # 3. LOGO and D1 vs D2
        l_c1, l_c2 = st.columns(2)
        with l_c1:
            st.markdown("### 🧩 Leave-One-Group-Out (LOGO) Ablation")
            st.markdown("Measures resilience and marginal loss when an entire feature category is eliminated from the full model:")
            if not df_logo.empty:
                st.plotly_chart(create_logo_ablation_chart(df_logo), use_container_width=True)
            else:
                st.info("LOGO table not available.")
                
        with l_c2:
            st.markdown("### ⚔️ Dataset 1 vs Dataset 2 Head-to-Head")
            st.markdown("Comparison of test set ROC-AUC across identical estimators fitted on 14 baseline vs 29 enhanced features:")
            if not df_d1_d2.empty:
                fig_d1_d2 = px.bar(
                    df_d1_d2,
                    x="Model",
                    y="ROC_AUC",
                    color="Dataset",
                    barmode="group",
                    text_auto=".4f",
                    color_discrete_map={
                        "Dataset 1 (Base 14 Features)": COLORS["blue"],
                        "Dataset 2 (Enhanced 29 Features)": COLORS["emerald"]
                    },
                    title="<b>D1 (14) vs D2 (29) ROC-AUC Comparison</b>"
                )
                fig_d1_d2.update_layout(**get_plotly_layout(height=380))
                fig_d1_d2.update_yaxes(range=[0.89, 0.93])
                st.plotly_chart(fig_d1_d2, use_container_width=True)
            else:
                st.info("D1 vs D2 table not available.")

    # =========================================================================
    # TAB 2: NCRB DYNAMIC FEATURES & 5-SEED ABLATION
    # =========================================================================
    with fa_tab2:
        ncrb_tables = load_ncrb_tables()
        df_ablation_seed = ncrb_tables.get("multiseed_ablation", pd.DataFrame())
        
        st.markdown("### ⏱️ Dynamic Longitudinal Features in NCRB Panel")
        st.markdown("""
        In addition to the 6 core IPC crime group counts, five longitudinal dynamic indicators were engineered to capture 
        crime volatility, specialization, and momentum over consecutive years:
        """)
        
        # Dynamic Features Table
        dyn_features = [
            {"Feature Name": "TOTAL_GROUPED_CRIME", "Mathematical Definition": "Sum of 6 core crime groups", "Criminological Rationale": "Measures district aggregate violent/property footprint independent of jurisdiction scale."},
            {"Feature Name": "ACTIVE_CRIME_GROUPS", "Mathematical Definition": "Count of groups with count > 0 (1 to 6)", "Criminological Rationale": "Measures whether criminality is multi-faceted or concentrated in a single offense type."},
            {"Feature Name": "CRIME_DIVERSITY_INDEX", "Mathematical Definition": "- sum(p_i * ln(p_i)) [Shannon Entropy]", "Criminological Rationale": "High entropy indicates diversified crime; low entropy indicates specialization (e.g. pure property or riots)."},
            {"Feature Name": "PREV_YEAR_TOTAL_CRIME", "Mathematical Definition": "Lagged Total Crime (t - 1)", "Criminological Rationale": "Captures historical volume baseline available prior to the forecast window."},
            {"Feature Name": "YOY_TOTAL_CRIME_CHANGE", "Mathematical Definition": "Total(t) - Total(t-1)", "Criminological Rationale": "Longitudinal momentum / annual rate of change in reported incidents."}
        ]
        st.dataframe(pd.DataFrame(dyn_features), use_container_width=True, hide_index=True)
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("### 🔬 5-Seed Paired Feature Ablation (GRU vs LSTM)")
        st.markdown("""
        To verify whether engineered features provide statistically consistent gains rather than optimization artifacts, 
        each dynamic feature was independently added to the baseline sequence across 5 random seeds (11, 23, 37, 53, 71):
        """)
        
        if not df_ablation_seed.empty:
            # Aggregate per feature and model
            agg_ablation = df_ablation_seed.groupby(["feature_added", "model"])["paired_macro_f1_difference"].agg(
                Mean_Diff="mean",
                Std_Diff="std"
            ).reset_index()
            
            fig_ncrb_abl = px.bar(
                agg_ablation,
                x="feature_added",
                y="Mean_Diff",
                color="model",
                barmode="group",
                text_auto=".4f",
                color_discrete_map={"GRU": COLORS["teal"], "LSTM": COLORS["emerald"]},
                title="<b>Mean Paired Macro F1 Difference Across 5 Seeds (Feature - Baseline)</b>"
            )
            fig_ncrb_abl.update_layout(**get_plotly_layout(height=400))
            fig_ncrb_abl.update_traces(textposition="outside")
            st.plotly_chart(fig_ncrb_abl, use_container_width=True)
            
            with st.expander("📋 View Multi-Seed Feature Ablation Data (Seeds 11, 23, 37, 53, 71)", expanded=False):
                st.dataframe(df_ablation_seed, use_container_width=True, hide_index=True)
        else:
            st.info("Multi-seed ablation table not available.")
