"""
Module C: Data Processing & Methodology.
Details end-to-end data pipelines, leakage prevention audits, and feature set architectures.
"""
import streamlit as st
import pandas as pd
from utils.config import COLORS

def render_methodology():
    """Renders Section C: Data Processing and Methodology."""
    st.markdown("""
    <div class="header-banner">
        <span class="badge-pill badge-teal">RIGOROUS DATA ENGINEERING</span>
        <h2 style="margin: 6px 0 2px 0; color: #f8fafc;">Data Processing & Leakage-Free Methodology</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
            How raw police dispatch logs and administrative crime records were systematically transformed into 
            leakage-free, model-ready feature spaces without introducing look-ahead bias or post-investigation artifacts.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    m_tab1, m_tab2 = st.tabs(["🏙️ Chicago Pipeline Methodology", "🗺️ Indian NCRB Pipeline Methodology"])
    
    # =========================================================================
    # TAB 1: CHICAGO METHODOLOGY
    # =========================================================================
    with m_tab1:
        st.markdown("### 🛡️ Chicago Target Leakage & Feature Audit")
        st.markdown("""
        In real-time police dispatch, when a 911 call or initial field report is registered, only pre-incident and dispatch-time attributes are known. 
        Using attributes populated *after* an officer investigates introduces severe **Target Leakage**, creating artificially inflated test accuracy that collapses in real-world deployment.
        """)
        
        # Leakage Audit Table
        leakage_data = [
            {"Field": "Description", "Reason for Exclusion": "Contains strings like 'DOMESTIC BATTERY SIMPLE' in 143k+ rows. Direct target leakage.", "Severity": "Critical Leak"},
            {"Field": "IUCR / FBI Code", "Reason for Exclusion": "Sub-codes directly distinguish domestic vs non-domestic statutory charges (e.g., IUCR 0486 = Domestic Battery).", "Severity": "Critical Leak"},
            {"Field": "Arrest", "Reason for Exclusion": "Post-investigation outcome flag. Occurs hours after incident occurrence; not dispatch-time info.", "Severity": "Post-Event Artifact"},
            {"Field": "Updated On", "Reason for Exclusion": "Maintenance timestamp written by police records management system days/weeks after the event.", "Severity": "Look-Ahead Leak"},
            {"Field": "Block", "Reason for Exclusion": "~35,000 unique text strings. Enables deep trees to memorize specific house numbers rather than learning patterns.", "Severity": "High Overfit Risk"},
            {"Field": "ID / Case Number", "Reason for Exclusion": "Arbitrary administrative primary keys with zero causal or predictive signal.", "Severity": "Spurious Identifier"},
            {"Field": "X/Y Coordinate, Location", "Reason for Exclusion": "Deterministic duplicates of Latitude and Longitude.", "Severity": "Redundant"}
        ]
        df_leak = pd.DataFrame(leakage_data)
        st.dataframe(df_leak, use_container_width=True, hide_index=True)
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("### ⚖️ Dataset 1 (14 Baseline Features) vs Dataset 2 (29 Enhanced Features)")
        st.markdown("""
        To measure the exact empirical value of advanced domain engineering, two distinct feature sets were constructed from the leak-free raw features:
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="panel-card" style="border-left: 3px solid {COLORS['blue']};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: #f8fafc;">Dataset 1: Baseline Architecture (14 Features)</h4>
                    <span class="badge-pill badge-cyan">Baseline Configuration</span>
                </div>
                <p style="font-size: 0.85rem; color: #94a3b8;">
                    Consists strictly of raw temporal attributes, police administrative boundaries, and raw categorical codes:
                </p>
                <ul style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                    <li><b>Temporal (4):</b> <code>Hour</code>, <code>Month</code>, <code>DayOfWeek</code>, <code>IsWeekend</code></li>
                    <li><b>Spatial (6):</b> <code>District</code>, <code>Beat</code>, <code>Ward</code>, <code>Community Area</code>, <code>Latitude</code>, <code>Longitude</code></li>
                    <li><b>Categorical (2):</b> <code>Primary Type</code>, <code>Location Description</code> (Ordinal Encoded)</li>
                    <li><b>Metadata (2):</b> <code>Date</code>, <code>split</code> (train/val/test indicator)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="panel-card" style="border-left: 3px solid {COLORS['emerald']};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: #f8fafc;">Dataset 2: Enhanced Architecture (29 Features)</h4>
                    <span class="badge-pill badge-emerald">Advanced Domain Features</span>
                </div>
                <p style="font-size: 0.85rem; color: #94a3b8;">
                    Augments Dataset 1 with 15 engineered features designed from criminological domain knowledge:
                </p>
                <ul style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                    <li><b>Trigonometric Cyclical Encodings (6):</b> <code>Hour_Sin/Cos</code>, <code>Month_Sin/Cos</code>, <code>DayOfWeek_Sin/Cos</code></li>
                    <li><b>Premises & Shift Flags (4):</b> <code>Is_Residential</code>, <code>Is_Public_Space</code>, <code>Is_Night</code>, <code>Time_Period</code></li>
                    <li><b>Bayesian Empirical Risk Priors (3):</b> <code>Primary_Type_Domestic_Risk</code>, <code>Location_Domestic_Risk</code>, <code>Beat_Domestic_Risk</code></li>
                    <li><b>Spatial Exposure & Interactions (2):</b> <code>Beat_Historical_Incident_Volume</code>, <code>Type_Location_Interaction</code></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🔄 Chronological Split Strategy (No Future Leakage)")
        st.markdown(f"""
        <div style="background-color: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 16px;">
            <div style="display: flex; justify-content: space-between; text-align: center;">
                <div style="flex: 3; background-color: rgba(59, 130, 246, 0.15); border: 1px solid {COLORS['blue']}; padding: 12px; border-radius: 6px; margin-right: 8px;">
                    <b style="color: {COLORS['blue']};">TRAINING SET (388,307 records | 51.4%)</b><br>
                    <span style="font-size: 0.8rem; color: #94a3b8;">Jan 1, 2023 – June 30, 2024 (18 months)</span>
                </div>
                <div style="flex: 1.2; background-color: rgba(245, 158, 11, 0.15); border: 1px solid {COLORS['amber']}; padding: 12px; border-radius: 6px; margin-right: 8px;">
                    <b style="color: {COLORS['amber']};">VALIDATION (131,449 | 17.4%)</b><br>
                    <span style="font-size: 0.8rem; color: #94a3b8;">July 1, 2024 – Dec 31, 2024</span>
                </div>
                <div style="flex: 2; background-color: rgba(16, 185, 129, 0.15); border: 1px solid {COLORS['emerald']}; padding: 12px; border-radius: 6px;">
                    <b style="color: {COLORS['emerald']};">LOCKED TEST SET (236,329 | 31.2%)</b><br>
                    <span style="font-size: 0.8rem; color: #94a3b8;">Jan 1, 2025 – Dec 31, 2025 (Full Year)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: INDIAN NCRB METHODOLOGY
    # =========================================================================
    with m_tab2:
        st.markdown("### 📊 Indian NCRB Panel Harmonization & Target Formulation")
        st.markdown("""
        The Indian National Crime Records Bureau publishes district-level counts across hundreds of distinct Indian Penal Code (IPC) crime sections. 
        Raw tables suffer from frequent taxonomy changes, naming discrepancies, and extreme sparsity.
        """)
        
        nc1, nc2 = st.columns(2)
        with nc1:
            st.markdown(f"""
            <div class="panel-card">
                <h4 style="margin-top: 0; color: #f8fafc;">1. Consolidation into 6 Crime Groups</h4>
                <p style="font-size: 0.85rem; color: #94a3b8;">
                    Hundreds of disparate IPC subsections were standardized and consolidated into 6 canonical crime families:
                </p>
                <ol style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                    <li><b>GROUP_HOMICIDE:</b> Murder, Attempt to Murder, Culpable Homicide.</li>
                    <li><b>GROUP_SEXUAL_VIOLENCE:</b> Rape, Custodial Rape, Gang Rape.</li>
                    <li><b>GROUP_KIDNAPPING_ABDUCTION:</b> Women/Girls Kidnapping, Ransom, Abduction.</li>
                    <li><b>GROUP_ROBBERY_DACOITY:</b> Dacoity, Preparation for Dacoity, Robbery.</li>
                    <li><b>GROUP_PUBLIC_VIOLENCE:</b> Riots (Communal, Caste, Political, Agrarian, Student).</li>
                    <li><b>GROUP_PROPERTY_CRIME:</b> Arson, Burglary, Theft, Extortion.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
        with nc2:
            st.markdown(f"""
            <div class="panel-card">
                <h4 style="margin-top: 0; color: #f8fafc;">2. Frozen Quintile Target Boundaries</h4>
                <p style="font-size: 0.85rem; color: #94a3b8;">
                    The 5-tier classification target (<code>TARGET_VIOLENT_LEVEL</code>) was defined using quintile boundaries fitted 
                    <b>exclusively on 2001–2010 training observations</b> and permanently frozen:
                </p>
                <table style="width: 100%; font-size: 0.82rem; color: #cbd5e1; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid {COLORS['border']};"><th>Tier</th><th>Burden Threshold</th><th>Category Meaning</th></tr>
                    <tr><td><b>Tier 1</b></td><td>&le; 107.0</td><td>Lowest Violent Burden</td></tr>
                    <tr><td><b>Tier 2</b></td><td>107.1 – 237.0</td><td>Low-to-Moderate Burden</td></tr>
                    <tr><td><b>Tier 3</b></td><td>237.1 – 362.0</td><td>Moderate Violent Burden</td></tr>
                    <tr><td><b>Tier 4</b></td><td>362.1 – 563.0</td><td>Moderate-to-High Burden</td></tr>
                    <tr><td><b>Tier 5</b></td><td>&gt; 563.0</td><td>Highest Violent Burden</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("#### ⏳ 3-Year Rolling Lag Sequence Structure")
        st.markdown(f"""
        <div class="panel-card">
            <p style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;">
                For recurrent sequence forecasting (GRU and LSTM), each district requires an unbroken historical record of 3 consecutive years 
                (\(t-3, t-2, t-1\)) to forecast the ordered burden level at year \(t\):
            </p>
            <div style="background-color: {COLORS['surface_light']}; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 0.85rem; color: {COLORS['emerald']};">
                Input Matrix: X[district, t] = [ Features(t-3), Features(t-2), Features(t-1) ]  &rarr;  Target: Y[district, t]
            </div>
            <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 8px;">
                <b>Locked Test Period (2013–2014):</b> Only districts with valid consecutive historical data through 2010–2012 (for 2013 forecast) 
                and 2011–2013 (for 2014 forecast) form the shared 671-sequence test cohort.
            </p>
        </div>
        """, unsafe_allow_html=True)
