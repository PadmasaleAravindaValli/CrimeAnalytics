"""
Module I: Technical Dataset & Experiment Specifications.
Complete schemas, feature dictionaries, hyperparameters, and downloadable research archives.
"""
import streamlit as st
import pandas as pd
from utils.config import COLORS
from utils.data_loader import load_chicago_tables, load_ncrb_tables

def render_dataset_details():
    """Renders Section I: Technical Dataset & Experiment Details."""
    st.markdown("""
    <div class="header-banner">
        <span class="badge-pill badge-cyan">TECHNICAL SPECIFICATION</span>
        <h2 style="margin: 6px 0 2px 0; color: #f8fafc;">Dataset Schemas, Hyperparameters & Provenance</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
            Exhaustive documentation of feature definitions, optimization hyperparameters, training protocols, 
            and reproducible data splits.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    t_tab1, t_tab2, t_tab3 = st.tabs(["🏙️ Chicago Technical Specs", "🗺️ NCRB Technical Specs", "📥 Download Research Tables"])
    
    # =========================================================================
    # TAB 1: CHICAGO TECHNICAL SPECS
    # =========================================================================
    with t_tab1:
        st.markdown("### 📋 Chicago Dataset 2 Feature Dictionary (29 Features)")
        
        feature_dict = [
            {"Name": "Date", "Type": "Datetime", "Description": "Timestamp of incident occurrence (2023–2025). Used exclusively for temporal sorting."},
            {"Name": "Primary Type", "Type": "Categorical", "Description": "Statutory offense category (e.g. THEFT, BATTERY, ASSAULT). Ordinal encoded."},
            {"Name": "Location Description", "Type": "Categorical", "Description": "Physical premises of occurrence (e.g. RESIDENCE, APARTMENT, STREET)."},
            {"Name": "District", "Type": "Integer (1-25)", "Description": "Police administrative district (22 active districts in Chicago)."},
            {"Name": "Beat", "Type": "Integer", "Description": "Micro-level police patrol sector (~277 unique beats across the city)."},
            {"Name": "Hour / Month / DayOfWeek", "Type": "Integer", "Description": "Discrete calendar temporal indicators."},
            {"Name": "IsWeekend / Is_Night", "Type": "Binary (0/1)", "Description": "Indicator flags for weekend days and nighttime hours (20:00 to 06:00)."},
            {"Name": "Hour_Sin / Hour_Cos", "Type": "Float", "Description": "Trigonometric continuous cyclical encodings for diurnal time [sin(2*pi*h/24)]."},
            {"Name": "Month_Sin / Month_Cos", "Type": "Float", "Description": "Trigonometric seasonal cyclical encodings [sin(2*pi*m/12)]."},
            {"Name": "DayOfWeek_Sin / Cos", "Type": "Float", "Description": "Trigonometric weekly cyclical encodings [sin(2*pi*d/7)]."},
            {"Name": "Is_Residential", "Type": "Binary (0/1)", "Description": "Domain indicator: True if location is Residence, Apartment, Porch, Yard."},
            {"Name": "Is_Public_Space", "Type": "Binary (0/1)", "Description": "Domain indicator: True if location is Street, Sidewalk, Alley, Park."},
            {"Name": "Primary_Type_Domestic_Risk", "Type": "Float [0,1]", "Description": "Empirical Bayesian prior: Historical domestic rate by offense type (train set)."},
            {"Name": "Location_Domestic_Risk", "Type": "Float [0,1]", "Description": "Empirical Bayesian prior: Historical domestic rate by location premise (train set)."},
            {"Name": "Beat_Domestic_Risk", "Type": "Float [0,1]", "Description": "Empirical Bayesian prior: Historical domestic rate by micro-beat (train set)."},
            {"Name": "Beat_Historical_Incident_Volume", "Type": "Integer", "Description": "Historical total incident density recorded in that beat during training."},
            {"Name": "Type_Location_Interaction", "Type": "Categorical", "Description": "Concatenated interaction feature (e.g. 'BATTERY_APARTMENT')."},
            {"Name": "Domestic", "Type": "Binary Target", "Description": "Target variable (1 = Domestic altercation, 0 = Non-domestic incident)."}
        ]
        st.dataframe(pd.DataFrame(feature_dict), use_container_width=True, hide_index=True)
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Chicago Model Hyperparameters")
        
        hp_c1, hp_c2 = st.columns(2)
        with hp_c1:
            st.markdown(f"""
            <div class="panel-card">
                <h5 style="margin-top: 0; color: #f8fafc;">XGBoost Classifier (Champion)</h5>
                <ul style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                    <li><b>n_estimators:</b> 300 boosting rounds</li>
                    <li><b>max_depth:</b> 6 levels</li>
                    <li><b>learning_rate:</b> 0.05</li>
                    <li><b>subsample:</b> 0.80 (row bagging)</li>
                    <li><b>colsample_bytree:</b> 0.80 (feature bagging)</li>
                    <li><b>scale_pos_weight:</b> 4.417 (inverse class frequency ratio)</li>
                    <li><b>objective:</b> binary:logistic</li>
                    <li><b>eval_metric:</b> logloss, auc</li>
                    <li><b>decision_threshold:</b> 0.47 (tuned for 87.18% Recall)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with hp_c2:
            st.markdown(f"""
            <div class="panel-card">
                <h5 style="margin-top: 0; color: #f8fafc;">PyTorch Deep Tabular MLP</h5>
                <ul style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                    <li><b>architecture:</b> Linear(29&rarr;128) &rarr; Linear(128&rarr;64) &rarr; Linear(64&rarr;32) &rarr; 1</li>
                    <li><b>activations:</b> ReLU on all hidden layers</li>
                    <li><b>regularization:</b> BatchNorm1d + Dropout (0.3 on layer 1, 0.2 on layer 2)</li>
                    <li><b>optimizer:</b> Adam (lr=1e-3, weight_decay=1e-4)</li>
                    <li><b>batch_size:</b> 512 | <b>epochs:</b> 40 with early stopping</li>
                    <li><b>loss:</b> BCEWithLogitsLoss(pos_weight=torch.tensor([4.417]))</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: NCRB TECHNICAL SPECS
    # =========================================================================
    with t_tab2:
        st.markdown("### 📋 NCRB Dataset Schema & Dynamic Feature Formulae")
        
        ncrb_schema = [
            {"Column": "YEAR", "Type": "Integer", "Role": "Longitudinal temporal index (2001 to 2014)."},
            {"Column": "STATE", "Type": "String", "Role": "State / Union Territory administrative name (35 units)."},
            {"Column": "UNIT_NAME", "Type": "String", "Role": "Police district entity (~600 unique districts)."},
            {"Column": "GROUP_HOMICIDE", "Type": "Float", "Role": "IPC Murder, Attempt to Murder, Culpable Homicide counts."},
            {"Column": "GROUP_SEXUAL_VIOLENCE", "Type": "Float", "Role": "IPC Rape and Custodial Sexual Assault counts."},
            {"Column": "GROUP_KIDNAPPING_ABDUCTION", "Type": "Float", "Role": "IPC Kidnapping and Abduction counts."},
            {"Column": "GROUP_ROBBERY_DACOITY", "Type": "Float", "Role": "IPC Robbery, Dacoity, and Preparation counts."},
            {"Column": "GROUP_PUBLIC_VIOLENCE", "Type": "Float", "Role": "IPC Riots (Caste, Communal, Political, Student) counts."},
            {"Column": "GROUP_PROPERTY_CRIME", "Type": "Float", "Role": "IPC Arson, Burglary, Theft, Extortion counts."},
            {"Column": "TOTAL_GROUPED_CRIME", "Type": "Float", "Role": "Sum across all 6 core crime groups."},
            {"Column": "ACTIVE_CRIME_GROUPS", "Type": "Integer (1-6)", "Role": "Number of groups with non-zero incident reports."},
            {"Column": "CRIME_DIVERSITY_INDEX", "Type": "Float [0, ln(6)]", "Role": "Shannon Entropy of crime category distribution."},
            {"Column": "PREV_YEAR_TOTAL_CRIME", "Type": "Float", "Role": "Lagged total crime count from year t-1."},
            {"Column": "YOY_TOTAL_CRIME_CHANGE", "Type": "Float", "Role": "Total_Crime(t) - Total_Crime(t-1)."},
            {"Column": "TARGET_VIOLENT_LEVEL", "Type": "Ordinal (1-5)", "Role": "5-Tier violent crime burden target."}
        ]
        st.dataframe(pd.DataFrame(ncrb_schema), use_container_width=True, hide_index=True)
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("### ⚙️ NCRB Recurrent Sequence Hyperparameters")
        
        np_c1, np_c2 = st.columns(2)
        with np_c1:
            st.markdown(f"""
            <div class="panel-card">
                <h5 style="margin-top: 0; color: #f8fafc;">LSTM Sequence Forecaster</h5>
                <ul style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                    <li><b>input_dimension:</b> 6 (Baseline) | 11 (Enhanced)</li>
                    <li><b>sequence_length:</b> 3 consecutive years (t-3, t-2, t-1)</li>
                    <li><b>hidden_dimension:</b> 64 recurrent units</li>
                    <li><b>num_layers:</b> 2 stacked LSTM layers</li>
                    <li><b>dropout:</b> 0.20 between recurrent layers</li>
                    <li><b>optimizer:</b> Adam (lr=1e-3, lr_scheduler on plateau)</li>
                    <li><b>loss:</b> Sparse Categorical Cross-Entropy (5 classes)</li>
                    <li><b>random_seeds:</b> 11, 23, 37, 53, 71</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with np_c2:
            st.markdown(f"""
            <div class="panel-card">
                <h5 style="margin-top: 0; color: #f8fafc;">GRU Sequence Forecaster</h5>
                <ul style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                    <li><b>input_dimension:</b> 6 (Baseline) | 11 (Enhanced)</li>
                    <li><b>sequence_length:</b> 3 consecutive years (t-3, t-2, t-1)</li>
                    <li><b>hidden_dimension:</b> 64 recurrent units</li>
                    <li><b>num_layers:</b> 2 stacked GRU layers</li>
                    <li><b>dropout:</b> 0.20 between recurrent layers</li>
                    <li><b>optimizer:</b> Adam (lr=1e-3, lr_scheduler on plateau)</li>
                    <li><b>loss:</b> Sparse Categorical Cross-Entropy (5 classes)</li>
                    <li><b>random_seeds:</b> 11, 23, 37, 53, 71</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: DOWNLOAD RESEARCH TABLES
    # =========================================================================
    with t_tab3:
        st.markdown("### 📥 Download Empirical Research CSVs")
        st.markdown("""
        All experimental result tables generated by the project scripts can be exported for peer review, external validation, or thesis inclusion:
        """)
        
        chicago_tables = load_chicago_tables()
        ncrb_tables = load_ncrb_tables()
        
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.markdown("#### 🏙️ Chicago Police Pipeline Tables")
            for key, name in [
                ("benchmark_d2", "Dataset 2 Benchmark (dataset2_model_benchmark.csv)"),
                ("benchmark_d1", "Dataset 1 Benchmark (dataset1_model_benchmark.csv)"),
                ("ablation_progression", "7-Stage Progression (feature_ablation_progression.csv)"),
                ("fairness", "22-District Fairness Audit (district_fairness_audit.csv)"),
                ("calibration", "Probability Calibration (calibration_metrics.csv)"),
                ("shap", "SHAP Feature Importance (shap_feature_importance.csv)"),
                ("stacking", "Stacking Meta-Ensemble (stacking_ensemble_results.csv)")
            ]:
                df_item = chicago_tables.get(key, pd.DataFrame())
                if not df_item.empty:
                    st.download_button(
                        f"⬇️ {name}",
                        df_item.to_csv(index=False).encode('utf-8'),
                        f"{key}.csv",
                        "text/csv",
                        key=f"dl_c_{key}"
                    )
                    
        with d_c2:
            st.markdown("#### 🗺️ Indian NCRB Pipeline Tables")
            for key, name in [
                ("thesis_ready", "Thesis Final Benchmark (thesis_ready_final_results.csv)"),
                ("multiseed_metrics", "5-Seed Model Metrics (multiseed_per_seed_metrics.csv)"),
                ("multiseed_ablation", "5-Seed Feature Ablation (per_seed_ablation_metrics.csv)"),
                ("target_thresholds", "Quintile Cut Points (target_thresholds.csv)"),
                ("class_distribution", "Target Class Distribution (target_class_distribution_comparison.csv)")
            ]:
                df_item = ncrb_tables.get(key, pd.DataFrame())
                if not df_item.empty:
                    st.download_button(
                        f"⬇️ {name}",
                        df_item.to_csv(index=False).encode('utf-8'),
                        f"{key}.csv",
                        "text/csv",
                        key=f"dl_n_{key}"
                    )
