"""
KPI Card Components for Streamlit Dashboard.
Cleanly formatted HTML strings with zero indentation to prevent Markdown code block leaks.
"""
import streamlit as st

def render_kpi(title, value, subtitle=None, delta=None, delta_positive=True, badge=None, badge_type="emerald"):
    """Renders a single styled KPI card without Markdown indentation quirks."""
    delta_html = ""
    if delta:
        delta_class = "kpi-delta-positive" if delta_positive else "kpi-delta-negative"
        symbol = "↑" if delta_positive else "↓"
        delta_html = f'<div class="{delta_class}">{symbol} {delta}</div>'
        
    badge_html = ""
    if badge:
        badge_html = f'<span class="badge-pill badge-{badge_type}">{badge}</span>'
        
    subtext_html = ""
    if subtitle:
        subtext_html = f'<div class="kpi-subtext">{subtitle}</div>'
        
    html = (
        f'<div class="kpi-container">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start;">'
        f'<div class="kpi-title">{title}</div>'
        f'{badge_html}'
        f'</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">'
        f'{subtext_html}'
        f'{delta_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_kpi_row(cards_data):
    """
    Renders a row of KPI cards across equal Streamlit columns.
    cards_data: list of dicts with keys: title, value, subtitle, delta, delta_positive, badge, badge_type
    """
    n = len(cards_data)
    cols = st.columns(n)
    for i, data in enumerate(cards_data):
        with cols[i]:
            render_kpi(**data)
