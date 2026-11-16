"""
Data Loader Utility with caching for high performance.
Loads Chicago Police Department data and Indian NCRB crime data.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from utils.config import (
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    NCRB_DATA_DIR,
    NCRB_TABLES_DIR,
    NCRB_AUDITS_DIR,
    NCRB_ROOT,
)

@st.cache_data(show_spinner=False)
def load_chicago_sample(sample_size=100000, full=False):
    """
    Loads Chicago Dataset 2 (Enhanced 29 features).
    Uses a stratified/uniform sample for lightning fast dashboard responsiveness,
    or full dataset if full=True.
    """
    file_path = PROCESSED_DATA_DIR / "dataset2_enhanced_features.csv"
    if not file_path.exists():
        return pd.DataFrame()
    
    # Read core columns for data explorer
    cols = [
        "Date", "Primary Type", "Location Description", "District", "Beat",
        "Hour", "Month", "DayOfWeek", "IsWeekend", "Is_Night",
        "Is_Residential", "Is_Public_Space", "Domestic", "split",
        "Primary_Type_Domestic_Risk", "Location_Domestic_Risk", "Beat_Domestic_Risk"
    ]
    
    try:
        if full:
            df = pd.read_csv(file_path, usecols=cols)
        else:
            # Fast reservoir or skiprows sampling
            df = pd.read_csv(file_path, usecols=cols, nrows=sample_size * 2)
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month_Name"] = df["Date"].dt.strftime("%b")
        days = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        df["Day_Name"] = df["DayOfWeek"].map(days)
        df["Domestic_Label"] = df["Domestic"].map({1: "Domestic", 0: "Non-Domestic"})
        return df
    except Exception as e:
        print(f"Error loading Chicago data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_chicago_tables():
    """Loads all experimental results tables for Chicago Incident Classification."""
    tables = {}
    csv_mappings = {
        "benchmark_d1": TABLES_DIR / "dataset1_model_benchmark.csv",
        "benchmark_d2": TABLES_DIR / "dataset2_model_benchmark.csv",
        "ablation_progression": TABLES_DIR / "feature_ablation_progression.csv",
        "leave_one_group_out": TABLES_DIR / "leave_one_group_out_ablation.csv",
        "d1_vs_d2": TABLES_DIR / "ablation_d1_vs_d2.csv",
        "fairness": TABLES_DIR / "district_fairness_audit.csv",
        "calibration": TABLES_DIR / "calibration_metrics.csv",
        "shap": TABLES_DIR / "shap_feature_importance.csv",
        "ranking": TABLES_DIR / "feature_comparison_ranking.csv",
        "stacking": TABLES_DIR / "stacking_ensemble_results.csv",
    }
    
    for key, path in csv_mappings.items():
        if path.exists():
            try:
                tables[key] = pd.read_csv(path)
            except Exception as e:
                print(f"Error loading {key}: {e}")
                tables[key] = pd.DataFrame()
        else:
            tables[key] = pd.DataFrame()
            
    return tables

@st.cache_data(show_spinner=False)
def load_ncrb_dataset():
    """Loads Indian NCRB district-level crime dataset (9,856 rows, 15 columns)."""
    file_path = NCRB_DATA_DIR / "district_enhanced_features.csv"
    if not file_path.exists():
        # Fallback to grouped reduced
        alt = NCRB_DATA_DIR / "district_grouped_reduced.csv"
        if alt.exists():
            file_path = alt
        else:
            return pd.DataFrame()
            
    try:
        df = pd.read_csv(file_path)
        # Class labels mapping
        class_labels = {
            1: "1: Lowest Burden (<=107)",
            2: "2: Low-Moderate (108-237)",
            3: "3: Moderate (238-362)",
            4: "4: Moderate-High (363-563)",
            5: "5: Highest Burden (>563)"
        }
        if "TARGET_VIOLENT_LEVEL" in df.columns:
            df["Burden_Label"] = df["TARGET_VIOLENT_LEVEL"].map(class_labels)
        return df
    except Exception as e:
        print(f"Error loading NCRB dataset: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_ncrb_tables():
    """Loads all experimental results tables for Indian NCRB Crime Burden Forecasting."""
    tables = {}
    csv_mappings = {
        "thesis_ready": NCRB_TABLES_DIR / "thesis_ready_final_results.csv",
        "all_models": NCRB_TABLES_DIR / "all_models_final_comparison.csv",
        "lstm_results": NCRB_TABLES_DIR / "lstm_baseline_vs_enhanced_results.csv",
        "gru_results": NCRB_TABLES_DIR / "gru_baseline_vs_enhanced_results.csv",
        "mlp_results": NCRB_TABLES_DIR / "mlp_baseline_vs_enhanced_results.csv",
        "ranking": NCRB_TABLES_DIR / "final_model_ranking.csv",
        "target_thresholds": NCRB_TABLES_DIR / "target_thresholds.csv",
        "class_distribution": NCRB_TABLES_DIR / "target_class_distribution_comparison.csv",
        "multiseed_metrics": NCRB_AUDITS_DIR / "multiseed_per_seed_metrics.csv",
    }
    
    for key, path in csv_mappings.items():
        if path.exists():
            try:
                tables[key] = pd.read_csv(path)
            except Exception as e:
                print(f"Error loading NCRB {key}: {e}")
                tables[key] = pd.DataFrame()
        else:
            tables[key] = pd.DataFrame()
            
    # Also load per_seed_ablation_metrics if available
    ablation_p = NCRB_ROOT / "ML_Implementation" / "corrected_temporal_results" / "corrected_v5_feature_ablation_multiseed" / "per_seed_ablation_metrics.csv"
    if ablation_p.exists():
        tables["multiseed_ablation"] = pd.read_csv(ablation_p)
    else:
        tables["multiseed_ablation"] = pd.DataFrame()
        
    return tables
