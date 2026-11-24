"""
Configuration and constants for Crime Analytics & ML Research Dashboard.
Dynamically locates project root and sets design palette.
"""
from pathlib import Path
import os

# Detect Project Root
CURRENT_DIR = Path(__file__).resolve().parent
CANDIDATE_ROOTS = [
    CURRENT_DIR.parent,
    Path(r"C:\Users\aru12\OneDrive\Desktop\capstone project"),
    Path(os.getcwd()),
]

PROJECT_ROOT = CURRENT_DIR.parent
for cand in CANDIDATE_ROOTS:
    if (cand / "outputs" / "tables").exists():
        PROJECT_ROOT = cand
        break

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MAPS_DIR = OUTPUTS_DIR / "maps"

# NCRB Reference Directory
NCRB_ROOT = PROJECT_ROOT / "reference_materials" / "Major-project-github-ref"
if not NCRB_ROOT.exists():
    # Check alternate location
    alt_ncrb = Path(r"C:\Users\aru12\OneDrive\Desktop\capstone project\reference_materials\Major-project-github-ref")
    if alt_ncrb.exists():
        NCRB_ROOT = alt_ncrb

NCRB_TABLES_DIR = NCRB_ROOT / "outputs" / "tables"
NCRB_DATA_DIR = NCRB_ROOT / "data" / "processed"
NCRB_AUDITS_DIR = NCRB_ROOT / "outputs" / "audits" / "final_corrections_v5"

# Design Palette - Dark Research Intelligence
COLORS = {
    "bg_dark": "#0b0f19",
    "surface": "#151c2c",
    "surface_light": "#1e293b",
    "border": "#232f48",
    "border_highlight": "#334155",
    "emerald": "#10b981",
    "teal": "#14b8a6",
    "cyan": "#06b6d4",
    "blue": "#3b82f6",
    "indigo": "#6366f1",
    "purple": "#8b5cf6",
    "amber": "#f59e0b",
    "rose": "#f43f5e",
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
}

PLOTLY_TEMPLATE = "plotly_dark"
COLORWAY = [
    COLORS["emerald"],
    COLORS["teal"],
    COLORS["cyan"],
    COLORS["blue"],
    COLORS["indigo"],
    COLORS["amber"],
    COLORS["rose"],
    COLORS["purple"],
]
