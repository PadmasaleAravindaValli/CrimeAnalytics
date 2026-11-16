"""
Theme and CSS Styling for Dark Research Intelligence Dashboard.
"""
import streamlit as st
import plotly.graph_objects as go
from utils.config import COLORS, COLORWAY

def apply_custom_theme():
    """Injects high-end Command Center CSS styles into Streamlit."""
    custom_css = f"""
    <style>
        /* Base page background */
        .stApp {{
            background-color: {COLORS['bg_dark']};
            color: {COLORS['text_primary']};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: #0d1322 !important;
            border-right: 1px solid {COLORS['border']};
        }}
        
        /* Metric card container */
        .kpi-container {{
            background-color: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 16px 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            transition: all 0.2s ease-in-out;
            height: 100%;
        }}
        .kpi-container:hover {{
            border-color: {COLORS['emerald']};
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.12);
        }}
        .kpi-title {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLORS['text_muted']};
            margin-bottom: 6px;
            font-weight: 600;
        }}
        .kpi-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: {COLORS['text_primary']};
            line-height: 1.2;
        }}
        .kpi-subtext {{
            font-size: 0.8rem;
            color: {COLORS['text_secondary']};
            margin-top: 6px;
        }}
        .kpi-delta-positive {{
            color: {COLORS['emerald']};
            font-weight: 600;
            font-size: 0.85rem;
        }}
        .kpi-delta-negative {{
            color: {COLORS['rose']};
            font-weight: 600;
            font-size: 0.85rem;
        }}
        
        /* Badge Pills */
        .badge-pill {{
            display: inline-block;
            padding: 3px 10px;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 9999px;
            letter-spacing: 0.04em;
        }}
        .badge-emerald {{
            background-color: rgba(16, 185, 129, 0.15);
            color: {COLORS['emerald']};
            border: 1px solid rgba(16, 185, 129, 0.35);
        }}
        .badge-teal {{
            background-color: rgba(20, 184, 166, 0.15);
            color: {COLORS['teal']};
            border: 1px solid rgba(20, 184, 166, 0.35);
        }}
        .badge-cyan {{
            background-color: rgba(6, 182, 212, 0.15);
            color: {COLORS['cyan']};
            border: 1px solid rgba(6, 182, 212, 0.35);
        }}
        .badge-amber {{
            background-color: rgba(245, 158, 11, 0.15);
            color: {COLORS['amber']};
            border: 1px solid rgba(245, 158, 11, 0.35);
        }}
        .badge-rose {{
            background-color: rgba(244, 63, 94, 0.15);
            color: {COLORS['rose']};
            border: 1px solid rgba(244, 63, 94, 0.35);
        }}
        
        /* Analytical Card Section */
        .panel-card {{
            background-color: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        /* Header Banner */
        .header-banner {{
            background: linear-gradient(135deg, rgba(21, 28, 44, 0.95) 0%, rgba(11, 15, 25, 0.95) 100%);
            border-left: 4px solid {COLORS['emerald']};
            border-radius: 6px;
            padding: 18px 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }}
        
        /* Custom Table Styling */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            background-color: {COLORS['surface']};
        }}
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent !important;
            border-radius: 4px !important;
            color: {COLORS['text_secondary']} !important;
            font-weight: 500 !important;
            padding: 8px 16px !important;
            border: 1px solid transparent !important;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {COLORS['surface_light']} !important;
            color: {COLORS['emerald']} !important;
            border: 1px solid {COLORS['border_highlight']} !important;
            font-weight: 600 !important;
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {COLORS['surface']} !important;
            border: 1px solid {COLORS['border']} !important;
            border-radius: 6px !important;
            color: {COLORS['text_primary']} !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def get_plotly_layout(title="", height=400, show_legend=True):
    """Returns a unified dark research layout dictionary for Plotly charts."""
    return dict(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(color=COLORS["text_primary"], size=15),
            x=0.01,
            y=0.97
        ),
        template="plotly_dark",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        height=height,
        margin=dict(l=45, r=30, t=50 if title else 25, b=45),
        colorway=COLORWAY,
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', color=COLORS["text_secondary"]),
        hoverlabel=dict(bgcolor=COLORS["surface_light"], font_size=12, font_color=COLORS["text_primary"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ) if show_legend else dict(visible=False),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["border"],
            zeroline=False,
            tickfont=dict(color=COLORS["text_secondary"], size=11),
            title=dict(font=dict(color=COLORS["text_primary"], size=12)),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS["border"],
            zeroline=False,
            tickfont=dict(color=COLORS["text_secondary"], size=11),
            title=dict(font=dict(color=COLORS["text_primary"], size=12)),
        ),
    )
