"""
Module B: Interactive Crime Data Explorer.
Provides dynamic filtering and rich visualization for both Chicago and Indian NCRB datasets.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_chicago_sample, load_ncrb_dataset
from utils.config import COLORS, COLORWAY
from components.theme import get_plotly_layout
from components.kpi_cards import render_kpi_row

def render_explorer():
    """Renders Section B: Crime Data Explorer."""
    st.markdown("""
    <div class="header-banner">
        <span class="badge-pill badge-cyan">INTERACTIVE DATA EXPLORER</span>
        <h2 style="margin: 6px 0 2px 0; color: #f8fafc;">Empirical Crime Dataset Exploration</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
            Filter, inspect distributions, analyze temporal trends, and explore spatial variance across Chicago Police Department incidents (2023–2025) 
            and National Crime Records Bureau district panels (2001–2014).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_chicago, tab_ncrb = st.tabs(["🏙️ Chicago Incident Explorer (CPD 2023–2025)", "🗺️ Indian NCRB District Panel (2001–2014)"])
    
    # =========================================================================
    # TAB 1: CHICAGO EXPLORER
    # =========================================================================
    with tab_chicago:
        df_chicago = load_chicago_sample(sample_size=100000)
        
        if df_chicago.empty:
            st.warning("Chicago processed dataset not found in `data/processed/dataset2_enhanced_features.csv`.")
        else:
            st.markdown("#### 🔍 Filter Chicago Police Incident Records")
            
            # Interactive Filter Bar
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                years = sorted(df_chicago["Year"].dropna().unique().tolist())
                sel_years = st.multiselect("Filter Year", years, default=years)
            with fc2:
                top_types = df_chicago["Primary Type"].value_counts().head(12).index.tolist()
                sel_types = st.multiselect("Crime Type (Top 12)", top_types, default=[])
            with fc3:
                districts = sorted([int(d) for d in df_chicago["District"].dropna().unique()])
                sel_districts = st.multiselect("Police District", districts, default=[])
            with fc4:
                dom_filter = st.selectbox("Domestic Classification", ["All Incidents", "Domestic Only (Class 1)", "Non-Domestic Only (Class 0)"])
                
            # Filter Data
            filtered_c = df_chicago.copy()
            if sel_years:
                filtered_c = filtered_c[filtered_c["Year"].isin(sel_years)]
            if sel_types:
                filtered_c = filtered_c[filtered_c["Primary Type"].isin(sel_types)]
            if sel_districts:
                filtered_c = filtered_c[filtered_c["District"].isin(sel_districts)]
            if dom_filter == "Domestic Only (Class 1)":
                filtered_c = filtered_c[filtered_c["Domestic"] == 1]
            elif dom_filter == "Non-Domestic Only (Class 0)":
                filtered_c = filtered_c[filtered_c["Domestic"] == 0]
                
            # Dynamic KPIs
            total_filtered = len(filtered_c)
            dom_count = int(filtered_c["Domestic"].sum())
            dom_rate = (dom_count / total_filtered * 100) if total_filtered > 0 else 0
            top_crime = filtered_c["Primary Type"].mode()[0] if not filtered_c.empty else "N/A"
            top_loc = filtered_c["Location Description"].mode()[0] if not filtered_c.empty else "N/A"
            
            kpi_data = [
                {"title": "Filtered Incidents", "value": f"{total_filtered:,}", "subtitle": "Sampled Explorer View", "badge": "Active Records", "badge_type": "cyan"},
                {"title": "Domestic Altercations", "value": f"{dom_count:,}", "subtitle": f"{dom_rate:.1f}% Prevalence", "badge": "Target Positive", "badge_type": "rose" if dom_rate > 20 else "teal"},
                {"title": "Most Frequent Crime", "value": top_crime, "subtitle": "Modal Category", "badge": "Primary Offense", "badge_type": "emerald"},
                {"title": "Top Location Type", "value": top_loc[:18], "subtitle": "Primary Venue", "badge": "Premises", "badge_type": "amber"}
            ]
            render_kpi_row(kpi_data)
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            # Charts Row 1: Temporal Trends
            r1_c1, r1_c2 = st.columns(2)
            with r1_c1:
                # Hourly distribution by domestic
                hour_counts = filtered_c.groupby(["Hour", "Domestic_Label"]).size().reset_index(name="Count")
                fig_hour = px.bar(
                    hour_counts,
                    x="Hour",
                    y="Count",
                    color="Domestic_Label",
                    barmode="stack",
                    color_discrete_map={"Domestic": COLORS["rose"], "Non-Domestic": COLORS["cyan"]},
                    title="<b>Incident Volume by Hour of Day</b>"
                )
                fig_hour.update_layout(**get_plotly_layout(height=350))
                st.plotly_chart(fig_hour, use_container_width=True)
                
            with r1_c2:
                # Day of week distribution
                day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                day_counts = filtered_c.groupby(["Day_Name", "Domestic_Label"]).size().reindex(day_order, level=0).reset_index(name="Count")
                fig_day = px.bar(
                    day_counts,
                    x="Day_Name",
                    y="Count",
                    color="Domestic_Label",
                    barmode="group",
                    color_discrete_map={"Domestic": COLORS["rose"], "Non-Domestic": COLORS["teal"]},
                    title="<b>Incident Volume by Day of Week</b>"
                )
                fig_day.update_layout(**get_plotly_layout(height=350))
                st.plotly_chart(fig_day, use_container_width=True)
                
            # Charts Row 2: Spatial & Venue Analysis
            r2_c1, r2_c2 = st.columns(2)
            with r2_c1:
                # District Domestic Rate vs Volume
                dist_agg = filtered_c.groupby("District").agg(
                    Total=("Domestic", "count"),
                    Domestic=("Domestic", "sum")
                ).reset_index()
                dist_agg["Domestic_Rate"] = dist_agg["Domestic"] / dist_agg["Total"] * 100
                dist_agg = dist_agg.sort_values("Domestic_Rate", ascending=False)
                
                fig_dist = px.bar(
                    dist_agg.head(15),
                    x="District",
                    y="Domestic_Rate",
                    color="Domestic_Rate",
                    color_continuous_scale=[[0, COLORS["teal"]], [1.0, COLORS["rose"]]],
                    title="<b>Top Districts by Domestic Violence Rate (%)</b>",
                    text_auto=".1f"
                )
                fig_dist.update_layout(**get_plotly_layout(height=350))
                fig_dist.update_traces(textposition="outside")
                fig_dist.update_coloraxes(showscale=False)
                st.plotly_chart(fig_dist, use_container_width=True)
                
            with r2_c2:
                # Top Venues
                venue_counts = filtered_c["Location Description"].value_counts().head(10).reset_index()
                venue_counts.columns = ["Location", "Count"]
                fig_venue = px.bar(
                    venue_counts,
                    x="Count",
                    y="Location",
                    orientation="h",
                    color="Count",
                    color_continuous_scale=[[0, COLORS["blue"]], [1.0, COLORS["emerald"]]],
                    title="<b>Top 10 Incident Location Descriptions</b>"
                )
                fig_venue.update_layout(**get_plotly_layout(height=350))
                fig_venue.update_layout(yaxis=dict(autorange="reversed"))
                fig_venue.update_coloraxes(showscale=False)
                st.plotly_chart(fig_venue, use_container_width=True)
                
            # Table View & Download
            with st.expander("📋 Inspect Filtered Incident Records (First 50 Rows) & Download CSV", expanded=False):
                disp_cols = ["Date", "Primary Type", "Location Description", "District", "Beat", "Hour", "Month", "Day_Name", "Domestic_Label", "Is_Residential"]
                st.dataframe(filtered_c[disp_cols].head(50), use_container_width=True)
                csv_bytes = filtered_c[disp_cols].head(5000).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Filtered Sample (CSV, up to 5,000 records)",
                    data=csv_bytes,
                    file_name="chicago_crime_filtered_sample.csv",
                    mime="text/csv"
                )

    # =========================================================================
    # TAB 2: INDIAN NCRB EXPLORER
    # =========================================================================
    with tab_ncrb:
        df_ncrb = load_ncrb_dataset()
        
        if df_ncrb.empty:
            st.warning("NCRB dataset not found in reference materials.")
        else:
            st.markdown("#### 🔍 Filter Indian National Crime Records Bureau (NCRB) District Panel")
            
            nfc1, nfc2, nfc3 = st.columns(3)
            with nfc1:
                states = sorted(df_ncrb["STATE"].dropna().unique().tolist())
                sel_states = st.multiselect("Filter State / UT", states, default=[])
            with nfc2:
                min_yr, max_yr = int(df_ncrb["YEAR"].min()), int(df_ncrb["YEAR"].max())
                sel_year_range = st.slider("Observation Year Range", min_value=min_yr, max_value=max_yr, value=(min_yr, max_yr))
            with nfc3:
                burdens = sorted(df_ncrb["TARGET_VIOLENT_LEVEL"].dropna().unique().tolist())
                sel_burdens = st.multiselect("Violent Burden Tier (1-5)", burdens, default=burdens)
                
            # Filter NCRB
            filtered_n = df_ncrb.copy()
            if sel_states:
                filtered_n = filtered_n[filtered_n["STATE"].isin(sel_states)]
            filtered_n = filtered_n[(filtered_n["YEAR"] >= sel_year_range[0]) & (filtered_n["YEAR"] <= sel_year_range[1])]
            if sel_burdens:
                filtered_n = filtered_n[filtered_n["TARGET_VIOLENT_LEVEL"].isin(sel_burdens)]
                
            # Dynamic KPIs
            n_dist_years = len(filtered_n)
            avg_tot_crime = filtered_n["TOTAL_GROUPED_CRIME"].mean() if "TOTAL_GROUPED_CRIME" in filtered_n.columns else 0
            avg_div_idx = filtered_n["CRIME_DIVERSITY_INDEX"].mean() if "CRIME_DIVERSITY_INDEX" in filtered_n.columns else 0
            
            # Find dominant crime group
            crime_group_cols = [
                "GROUP_HOMICIDE", "GROUP_SEXUAL_VIOLENCE", "GROUP_KIDNAPPING_ABDUCTION",
                "GROUP_ROBBERY_DACOITY", "GROUP_PUBLIC_VIOLENCE", "GROUP_PROPERTY_CRIME"
            ]
            group_sums = filtered_n[crime_group_cols].sum() if not filtered_n.empty else pd.Series()
            dominant_grp = group_sums.idxmax().replace("GROUP_", "").replace("_", " ") if not group_sums.empty else "N/A"
            
            ncrb_kpis = [
                {"title": "District-Year Records", "value": f"{n_dist_years:,}", "subtitle": f"{filtered_n['UNIT_NAME'].nunique()} Unique Districts", "badge": "Panel Depth", "badge_type": "cyan"},
                {"title": "Mean Annual Crime Volume", "value": f"{avg_tot_crime:.1f}", "subtitle": "Incidents per District-Year", "badge": "Grouped IPC", "badge_type": "teal"},
                {"title": "Dominant Crime Group", "value": dominant_grp, "subtitle": "Highest Cumulative Volume", "badge": "Dominant IPC", "badge_type": "emerald"},
                {"title": "Crime Diversity (Entropy)", "value": f"{avg_div_idx:.3f}", "subtitle": "Shannon Entropy Index", "badge": "Dispersion", "badge_type": "amber"}
            ]
            render_kpi_row(ncrb_kpis)
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            # Charts Row 1: Crime Group Trajectories & Burden Distribution
            nr1_c1, nr1_c2 = st.columns(2)
            with nr1_c1:
                # 6 groups over time
                yearly_groups = filtered_n.groupby("YEAR")[crime_group_cols].mean().reset_index()
                fig_groups = go.Figure()
                for c_col, col_name in zip(crime_group_cols, COLORWAY[:6]):
                    fig_groups.add_trace(go.Scatter(
                        x=yearly_groups["YEAR"],
                        y=yearly_groups[c_col],
                        mode="lines+markers",
                        name=c_col.replace("GROUP_", "").replace("_", " ").title(),
                        line=dict(color=col_name, width=2.5)
                    ))
                fig_groups.update_layout(**get_plotly_layout(title="<b>Mean Incidents per District by Crime Group (2001–2014)</b>", height=370))
                fig_groups.update_layout(xaxis=dict(title="Year", tickmode="linear", dtick=2))
                st.plotly_chart(fig_groups, use_container_width=True)
                
            with nr1_c2:
                # Target Burden Tier Distribution
                tier_counts = filtered_n["TARGET_VIOLENT_LEVEL"].value_counts().sort_index().reset_index()
                tier_counts.columns = ["Tier", "Count"]
                tier_names = {1: "1: <=107 (Lowest)", 2: "2: 108-237", 3: "3: 238-362", 4: "4: 363-563", 5: "5: >563 (Highest)"}
                tier_counts["Tier_Label"] = tier_counts["Tier"].map(tier_names)
                
                fig_tier = px.bar(
                    tier_counts,
                    x="Tier_Label",
                    y="Count",
                    color="Tier",
                    color_continuous_scale=[[0, COLORS["teal"]], [0.5, COLORS["amber"]], [1.0, COLORS["rose"]]],
                    title="<b>Target Violent Crime Burden Distribution (5 Quintile Tiers)</b>",
                    text_auto=True
                )
                fig_tier.update_layout(**get_plotly_layout(height=370))
                fig_tier.update_coloraxes(showscale=False)
                fig_tier.update_traces(textposition="outside")
                st.plotly_chart(fig_tier, use_container_width=True)
                
            # Charts Row 2: State Volume & Diversity Index
            nr2_c1, nr2_c2 = st.columns(2)
            with nr2_c1:
                # Top States by Total Grouped Crime
                state_agg = filtered_n.groupby("STATE")["TOTAL_GROUPED_CRIME"].mean().sort_values(ascending=False).head(10).reset_index()
                fig_state = px.bar(
                    state_agg,
                    x="TOTAL_GROUPED_CRIME",
                    y="STATE",
                    orientation="h",
                    color="TOTAL_GROUPED_CRIME",
                    color_continuous_scale=[[0, COLORS["blue"]], [1.0, COLORS["rose"]]],
                    title="<b>Top 10 States by Mean District Crime Volume</b>",
                    text_auto=".1f"
                )
                fig_state.update_layout(**get_plotly_layout(height=360))
                fig_state.update_layout(yaxis=dict(autorange="reversed"))
                fig_state.update_coloraxes(showscale=False)
                st.plotly_chart(fig_state, use_container_width=True)
                
            with nr2_c2:
                # Scatter: Total Crime vs YoY Change
                fig_scatter = px.scatter(
                    filtered_n.sample(n=min(1500, len(filtered_n)), random_state=42) if len(filtered_n) > 1500 else filtered_n,
                    x="TOTAL_GROUPED_CRIME",
                    y="YOY_TOTAL_CRIME_CHANGE",
                    color="TARGET_VIOLENT_LEVEL",
                    color_continuous_scale=[[0, COLORS["teal"]], [1.0, COLORS["rose"]]],
                    title="<b>Crime Volume vs Year-over-Year Volatility</b>",
                    hover_data=["STATE", "UNIT_NAME", "YEAR"]
                )
                fig_scatter.update_layout(**get_plotly_layout(height=360))
                fig_scatter.update_coloraxes(showscale=False)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            # Table View & Download
            with st.expander("📋 Inspect Filtered NCRB Panel Records & Download CSV", expanded=False):
                st.dataframe(filtered_n.head(50), use_container_width=True)
                csv_bytes_n = filtered_n.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Filtered NCRB Data (CSV)",
                    data=csv_bytes_n,
                    file_name="ncrb_district_filtered.csv",
                    mime="text/csv"
                )
