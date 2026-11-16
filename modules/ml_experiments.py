"""
Module E: Machine Learning Experiment Laboratory.
Interactive multi-metric comparison, threshold analysis, and model benchmarking.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_chicago_tables, load_ncrb_tables
from utils.config import COLORS, COLORWAY
from components.theme import get_plotly_layout
from components.charts import create_model_comparison_bar

def render_ml_experiments():
    """Renders Section E: Machine Learning Experiment Laboratory."""
    st.markdown("""
    <div class="header-banner">
        <span class="badge-pill badge-indigo">EXPERIMENTAL BENCHMARK</span>
        <h2 style="margin: 6px 0 2px 0; color: #f8fafc;">Machine Learning Experiment Laboratory</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
            Examine empirical benchmark results, evaluate threshold tuning trade-offs, and compare classical machine learning 
            with ensemble architectures across held-out test splits.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    lab_tab1, lab_tab2 = st.tabs(["🏙️ Chicago Model Laboratory", "🗺️ Indian NCRB Model Laboratory"])
    
    # =========================================================================
    # TAB 1: CHICAGO ML LABORATORY
    # =========================================================================
    with lab_tab1:
        chicago_tables = load_chicago_tables()
        df_d1 = chicago_tables.get("benchmark_d1", pd.DataFrame())
        df_d2 = chicago_tables.get("benchmark_d2", pd.DataFrame())
        df_stacking = chicago_tables.get("stacking", pd.DataFrame())
        
        st.markdown("### 🏆 Chicago Model Benchmark Comparison")
        st.markdown("""
        All models evaluated on the identical 2025 Test Cohort (236,329 incidents). Notice how tree ensembles (XGBoost) 
        dominate static tabular data, reaching **0.9152 ROC-AUC** and **87.18% Recall**.
        """)
        
        # Select dataset and metric
        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            dataset_choice = st.radio(
                "Select Feature Architecture:",
                ["Dataset 2 (Enhanced 29 Features)", "Dataset 1 (Baseline 14 Features)", "Both (Comparative View)"],
                horizontal=True
            )
        with col_ctrl2:
            metrics_available = ["ROC_AUC", "Balanced_Accuracy", "Recall", "F1_Score", "Precision", "Accuracy"]
            metric_selected = st.selectbox("Select Target Evaluation Metric:", metrics_available, index=0)
            
        # Prepare Plot Data
        if dataset_choice == "Dataset 2 (Enhanced 29 Features)":
            plot_df = df_d2.copy()
            if not plot_df.empty:
                st.plotly_chart(
                    create_model_comparison_bar(plot_df, metric_col=metric_selected, title=f"Dataset 2 (Enhanced) - {metric_selected}"),
                    use_container_width=True
                )
                st.dataframe(plot_df.sort_values(metric_selected, ascending=False), use_container_width=True, hide_index=True)
        elif dataset_choice == "Dataset 1 (Baseline 14 Features)":
            plot_df = df_d1.copy()
            if not plot_df.empty:
                st.plotly_chart(
                    create_model_comparison_bar(plot_df, metric_col=metric_selected, title=f"Dataset 1 (Baseline) - {metric_selected}"),
                    use_container_width=True
                )
                st.dataframe(plot_df.sort_values(metric_selected, ascending=False), use_container_width=True, hide_index=True)
        else:
            # Combined
            combined_df = []
            if not df_d1.empty:
                d1_copy = df_d1.copy()
                d1_copy["Feature_Set"] = "Dataset 1 (14 Features)"
                combined_df.append(d1_copy)
            if not df_d2.empty:
                d2_copy = df_d2.copy()
                d2_copy["Feature_Set"] = "Dataset 2 (29 Features)"
                combined_df.append(d2_copy)
            if combined_df:
                merged = pd.concat(combined_df, ignore_index=True)
                st.plotly_chart(
                    create_model_comparison_bar(merged, metric_col=metric_selected, group_col="Feature_Set", title=f"D1 vs D2 - {metric_selected}"),
                    use_container_width=True
                )
                st.dataframe(merged.sort_values([metric_selected, "Model"], ascending=[False, True]), use_container_width=True, hide_index=True)
                
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        # Stacking Meta-Ensemble Analysis
        st.markdown("### 🥞 Stacking Meta-Ensemble Architecture")
        st.markdown("""
        To evaluate if heterogeneous hypothesis spaces combine effectively, out-of-fold probability predictions from 
        Logistic Regression, HistGradientBoosting, and XGBoost were combined using an L2-penalized Logistic Regression meta-learner:
        """)
        
        if not df_stacking.empty:
            sc1, sc2 = st.columns([1.2, 1])
            with sc1:
                fig_stack = px.bar(
                    df_stacking,
                    x="Model",
                    y=["ROC_AUC", "Balanced_Acc", "Recall"],
                    barmode="group",
                    color_discrete_sequence=[COLORS["emerald"], COLORS["teal"], COLORS["cyan"]],
                    title="<b>Base Estimators vs Stacked Meta-Ensemble</b>"
                )
                fig_stack.update_layout(**get_plotly_layout(height=380))
                fig_stack.update_yaxes(range=[0.75, 0.95])
                st.plotly_chart(fig_stack, use_container_width=True)
            with sc2:
                st.dataframe(df_stacking, use_container_width=True, hide_index=True, height=380)
                st.info("💡 **Key Finding:** While the Stacked Meta-Ensemble matches the performance of the best base tree model (0.9122 ROC-AUC), it does not statistically outperform standalone XGBoost (0.9152 ROC-AUC), indicating tree-based gradient boosting already captures the dominant feature interactions.")
                
        # Download Chicago CSV
        if not df_d2.empty:
            csv_d2 = df_d2.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Dataset 2 Model Benchmark (CSV)", csv_d2, "chicago_d2_benchmark.csv", "text/csv")

    # =========================================================================
    # TAB 2: NCRB ML LABORATORY
    # =========================================================================
    with lab_tab2:
        ncrb_tables = load_ncrb_tables()
        df_thesis = ncrb_tables.get("thesis_ready", pd.DataFrame())
        df_ranking = ncrb_tables.get("ranking", pd.DataFrame())
        
        st.markdown("### 🗺️ NCRB Model Performance Across Splits")
        st.markdown("""
        Comparison of recurrent sequence architectures (GRU, LSTM) evaluated on chronological future-year sequences 
        versus traditional estimators evaluated on standard random splits:
        """)
        
        if not df_thesis.empty:
            st.dataframe(df_thesis, use_container_width=True, hide_index=True)
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            
            # Interactive Filter by Framework
            fw_choice = st.multiselect(
                "Filter Framework:",
                df_thesis["Framework"].unique().tolist(),
                default=df_thesis["Framework"].unique().tolist()
            )
            filtered_thesis = df_thesis[df_thesis["Framework"].isin(fw_choice)]
            
            metric_ncrb = st.selectbox("Select Metric to Visualize:", ["ROC_AUC", "F1", "Accuracy", "PR_AUC"], index=0)
            
            fig_ncrb_bench = px.bar(
                filtered_thesis,
                x="Model",
                y=metric_ncrb,
                color="Dataset",
                barmode="group",
                facet_col="Split",
                text_auto=".4f",
                color_discrete_map={"Baseline": COLORS["teal"], "Enhanced": COLORS["emerald"]},
                title=f"<b>NCRB Model Comparison by {metric_ncrb} across Splits</b>"
            )
            fig_ncrb_bench.update_layout(**get_plotly_layout(height=430))
            fig_ncrb_bench.update_traces(textposition="outside")
            st.plotly_chart(fig_ncrb_bench, use_container_width=True)
            
            st.warning("""
            ⚠️ **CRITICAL METHODOLOGICAL INSIGHT: The Random Split Trap**<br>
            Notice that the Random Stratified 80/20 MLP achieves an apparent **0.9056 Accuracy** and **0.9907 ROC-AUC**. 
            This is an illusion caused by **Target Leakage**: nine individual IPC crime count columns fed as inputs were simultaneously 
            used to construct the violent burden target in the exact same year. When evaluated on genuine future-year forecasting 
            using strict chronological sequences, true performance is **0.7674 Accuracy / 0.7708 F1 (LSTM)** and **0.7097 Macro F1**.
            """, icon="⚠️")
            
            csv_ncrb = df_thesis.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download NCRB Experimental Benchmark (CSV)", csv_ncrb, "ncrb_benchmark.csv", "text/csv")
        else:
            st.info("NCRB benchmark table not available.")
