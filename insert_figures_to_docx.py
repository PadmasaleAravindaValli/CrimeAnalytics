import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import os

docx_path = r'C:\Users\aru12\OneDrive\Desktop\CrimeAnalytics_paper.docx'
fig_dir = r'C:\Users\aru12\OneDrive\Desktop\travel\figures'

doc = docx.Document(docx_path)

# Helper function to add a centered figure with a caption right after a target paragraph
def insert_figure_after(doc, target_p, img_path, caption_text, width_inches=6.0):
    # Create new paragraph after target_p
    p_img = target_p.insert_paragraph_before() # We will insert before or after using XML
    # Actually python-docx doesn't have insert_paragraph_after directly, but we can do:
    new_p_img = OxmlElement('w:p')
    target_p._element.addnext(new_p_img)
    p_img = docx.text.paragraph.Paragraph(new_p_img, target_p._parent)
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(2)
    run_img = p_img.add_run()
    run_img.add_picture(img_path, width=Inches(width_inches))
    
    # Add caption paragraph after the image paragraph
    new_p_cap = OxmlElement('w:p')
    new_p_img.addnext(new_p_cap)
    p_cap = docx.text.paragraph.Paragraph(new_p_cap, target_p._parent)
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    run_cap = p_cap.add_run(caption_text)
    run_cap.font.name = 'Calibri'
    run_cap.font.size = Pt(9.5)
    run_cap.font.italic = True
    run_cap.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p_cap

# Map target locations
p_targets = {}
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if 'Algorithm 1:' in t:
        p_targets['fig1'] = p
    elif 'Table 3:' in t:
        p_targets['fig2'] = p
    elif 'Fig. 3b shows a Leave-One-Group-Out' in t:
        p_targets['fig3'] = p
    elif 'Table 5:' in t:
        p_targets['fig4'] = p
    elif 'Table 6:' in t:
        p_targets['fig7'] = p
    elif 'Table 7:' in t and 'Table 7b' not in t:
        p_targets['fig5'] = p
    elif 'Table 7b:' in t:
        p_targets['fig6'] = p

print("Found targets:", list(p_targets.keys()))

# 1. Figure 1: Overall System Architecture
if 'fig1' in p_targets:
    insert_figure_after(
        doc, p_targets['fig1'],
        os.path.join(fig_dir, 'fig1_overall_workflow.png'),
        "Fig. 1. Dual-pipeline methodology workflow: Chicago CPD report-time domestic incident classification (Pipeline A) and Indian NCRB district-level sequential violent crime forecasting (Pipeline B), operating under strict decision-support guardrails.",
        width_inches=6.2
    )
    print("Inserted Fig 1")

# 2. Figure 2: Chicago Model Performance & ROC
if 'fig2' in p_targets:
    insert_figure_after(
        doc, p_targets['fig2'],
        os.path.join(fig_dir, 'fig2_chicago_model_comparison.png'),
        "Fig. 2. Chicago CPD domestic violence classification performance comparison across architectures on Dataset 1 (Baseline) vs Dataset 2 (Enhanced), and ROC curves on the locked 2025 test cohort with calibrated operating threshold \u03c4* = 0.47.",
        width_inches=6.2
    )
    print("Inserted Fig 2")

# 3. Figure 3: 7-Stage Progression & LOGO Ablation
if 'fig3' in p_targets:
    insert_figure_after(
        doc, p_targets['fig3'],
        os.path.join(fig_dir, 'fig3_feature_progression_and_logo.png'),
        "Fig. 3. (a) 7-Stage cumulative ROC-AUC progression demonstrating the +0.2256 jump upon introducing offence categories. (b) Leave-One-Group-Out (LOGO) ablation showing degradation when feature groups are excluded, with \u2248 0 explicitly marked for cyclical and venue groups (Reviewer B7).",
        width_inches=6.2
    )
    print("Inserted Fig 3")

# 4. Figure 4: Probability Calibration
if 'fig4' in p_targets:
    insert_figure_after(
        doc, p_targets['fig4'],
        os.path.join(fig_dir, 'fig4_probability_calibration.png'),
        "Fig. 4. Probability calibration analysis on the 2025 test cohort: (a) Reliability diagram comparing uncalibrated XGBoost (ECE = 14.60% inflated by scale_pos_weight \u2248 4.43), Platt scaling (ECE = 2.15%), and Isotonic regression (ECE = 0.47%). (b) Score distribution realignment.",
        width_inches=6.2
    )
    print("Inserted Fig 4")

# 5. Figure 7: Algorithmic Fairness Audit Across 22 Districts
if 'fig7' in p_targets:
    insert_figure_after(
        doc, p_targets['fig7'],
        os.path.join(fig_dir, 'fig7_fairness_audit_districts.png'),
        "Fig. 7. 22-District algorithmic fairness audit on the 2025 test cohort: (a) Disparate impact ratio relative to reference District 24, with the EEOC 0.80\u20131.25 parity band highlighted. (b) Error rate disparities with Wilson 95% CIs, proving equalized odds is not achieved across all beats (Reviewer A5/B4).",
        width_inches=6.2
    )
    print("Inserted Fig 7")

# 6. Figure 5: NCRB Recurrent Models & 5-Seed Feature Ablation
if 'fig5' in p_targets:
    insert_figure_after(
        doc, p_targets['fig5'],
        os.path.join(fig_dir, 'fig5_ncrb_recurrent_comparison.png'),
        "Fig. 5. Indian NCRB sequence forecasting evaluation: (a) 5-Seed Macro F1 comparison (LSTM: 0.7097 \u00b1 0.0066 vs GRU: 0.7083 \u00b1 0.0076, paired t-test t = 0.345, p = 0.747, difference not statistically significant). (b) 5-Seed sequential feature ablation contribution (Reviewer B9).",
        width_inches=6.2
    )
    print("Inserted Fig 5")

# 7. Figure 6: Confusion Matrices
if 'fig6' in p_targets:
    insert_figure_after(
        doc, p_targets['fig6'],
        os.path.join(fig_dir, 'fig6_confusion_matrices.png'),
        "Fig. 6. Normalized confusion matrices: (a) Chicago CPD report-time domestic incident classifier on the locked 2025 test cohort (n = 236,329). (b) Indian NCRB enhanced LSTM 5-tier crime burden forecasting on the locked test partition (n = 671, Seed 42 closest to mean).",
        width_inches=6.2
    )
    print("Inserted Fig 6")

# Save document to all destinations
destinations = [
    r'C:\Users\aru12\OneDrive\Desktop\CrimeAnalytics_paper.docx',
    r'C:\Users\aru12\OneDrive\Desktop\CrimeAnalytics\CrimeAnalytics_paper.docx',
    r'C:\Users\aru12\OneDrive\Desktop\travel\CrimeAnalytics_paper.docx'
]

for d in destinations:
    try:
        doc.save(d)
        print(f"Saved successfully to: {d}")
    except Exception as e:
        print(f"Error saving to {d}: {e}")
