"""
Module H: Research Findings & Theoretical Insights.
In-depth academic discussion on data leakage, chronological splitting, class imbalance, and ML vs DL paradigms.
"""
import streamlit as st
from utils.config import COLORS

def render_research_insights():
    """Renders Section H: Research Findings and Insights."""
    st.markdown("""
    <div class="header-banner">
        <span class="badge-pill badge-rose">CRITICAL METHODOLOGICAL SYNTHESIS</span>
        <h2 style="margin: 6px 0 2px 0; color: #f8fafc;">Research Findings, Insights & Lessons Learned</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
            Rigorous examination of the methodological traps, empirical trade-offs, and public policy implications 
            discovered during the execution of both research pipelines.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 8 Major Insight Sections
    insights = [
        {
            "num": "01",
            "title": "The Data Leakage Trap: The Illusion of 90%+ Accuracy",
            "tag": "Data Integrity",
            "color": COLORS["rose"],
            "content": """
            In both pipelines, initial unconstrained models yielded deceptively high metrics that crumbled under forensic audit:
            <ul>
                <li><b>Chicago Leakage:</b> The police department's <code>Description</code> field literally recorded words like <code>DOMESTIC BATTERY SIMPLE</code> in 143,000+ incidents. Any model given raw text strings achieves trivial 99% accuracy by memorizing the target word. Furthermore, the <code>Arrest</code> flag is entered hours after the crime scene investigation, representing an operational look-ahead leak.</li>
                <li><b>NCRB Leakage:</b> In early uncorrected literature, an MLP achieved <b>90.76% accuracy and 0.9907 ROC-AUC</b> using an 80/20 random stratified split. Forensic column mapping revealed that nine IPC input features were algebraically part of the very violent crime burden used to construct the target. When re-evaluated on a leak-free 3-year chronological sequence, genuine forecasting accuracy is <b>76.7% / 0.7097 Macro F1</b>.</li>
            </ul>
            <b>Takeaway:</b> High benchmark scores in published crime literature must be viewed with extreme skepticism until feature temporal eligibility and target independence are forensically proven.
            """
        },
        {
            "num": "02",
            "title": "Why Chronological Splitting is Non-Negotiable",
            "tag": "Validation Rigor",
            "color": COLORS["amber"],
            "content": """
            Standard cross-validation (K-fold or random stratified splitting) implicitly allows models to train on records from Wednesday afternoon 
            to predict crimes on Tuesday morning in the same precinct. 
            Because crime exhibits localized temporal autocorrelation, random splits allow models to interpolate between concurrent events rather than forecasting genuine future distributions. 
            By locking the test periods strictly into the future (2025 for Chicago, 2013–2014 for NCRB), our framework measures genuine out-of-distribution temporal generalization.
            """
        },
        {
            "num": "03",
            "title": "Class Imbalance Reality: Accuracy is a Deceptive Vanity Metric",
            "tag": "Metric Selection",
            "color": COLORS["cyan"],
            "content": """
            Domestic violence Altercations represent <b>18.46%</b> of total Chicago crime records. 
            A naive 'dummy' classifier that predicts every single incident as Non-Domestic achieves <b>81.54% overall accuracy</b> while completely failing to detect a single domestic abuse survivor (0.00% Recall).
            <br><br>
            By optimizing for <b>Balanced Accuracy (83.55%)</b>, tuning the dispatch decision threshold to <b>0.47</b>, and monitoring the PR-AUC, 
            our champion XGBoost model achieved <b>87.18% Recall</b> (capturing 37,749 out of 43,300 domestic altercations in the 2025 test set) with 50.63% precision.
            """
        },
        {
            "num": "04",
            "title": "Tree Models vs Neural Networks: The Tabular vs Sequential Divide",
            "tag": "Architecture Match",
            "color": COLORS["emerald"],
            "content": """
            The dual pipeline provides a clear empirical verdict on the ML vs DL architectural debate:
            <ul>
                <li><b>For Static Incident Classification (Chicago):</b> XGBoost outperformed the PyTorch Deep MLP (0.9152 vs 0.9124 ROC-AUC) while training in 1/10th the computational time and being completely robust to unscaled inputs and hyperparameter sensitivity.</li>
                <li><b>For Longitudinal Burden Forecasting (NCRB):</b> Recurrent sequence models (LSTM, Macro F1: 0.7097) decisively outperformed flat tabular models (Logistic Regression, Macro F1: 0.6761) by more than +3.3 F1 points. Multi-year recurrence successfully captured long-term non-linear momentum across administrative cycles.</li>
            </ul>
            """
        },
        {
            "num": "05",
            "title": "The Transformative Power of Domain-Guided Feature Engineering",
            "tag": "Feature Science",
            "color": COLORS["teal"],
            "content": """
            The 7-stage ablation progression reveals that raw temporal indicators alone achieved an ROC-AUC of barely <b>0.5593</b> (slightly better than random coin toss). 
            Adding spatial coordinates improved discrimination to <b>0.6876</b>. 
            The breakthrough came with Bayesian empirical risk priors and residential venue indicators, catapulting performance to <b>0.9150+ ROC-AUC</b>. 
            Thoughtful feature engineering rooted in criminological domain reality yielded a <b>+0.3557 ROC-AUC improvement</b>, proving that feature representation matters far more than hyperparameter tuning.
            """
        },
        {
            "num": "06",
            "title": "Multi-Seed Experiments: Distinguishing Signal from Stochastic Noise",
            "tag": "Statistical Stability",
            "color": COLORS["purple"],
            "content": """
            In deep neural training, single-seed results frequently masquerade as algorithmic superiority. 
            Our 5-seed evaluation across Seeds 11, 23, 37, 53, and 71 proved that seed variance (\(\pm 0.007\)) can exceed the margin between competing architectures. 
            Reporting mean performance and paired difference standard deviations provides transparency and prevents overclaiming marginal architectural tweaks.
            """
        },
        {
            "num": "07",
            "title": "Algorithmic Fairness: Parity in Sensitivity Across Diverse Neighborhoods",
            "tag": "Ethical AI",
            "color": COLORS["blue"],
            "content": """
            Fairness audits are mandatory for algorithms informing emergency public services. 
            Auditing 22 Chicago Police Districts proved that the model maintains an equal opportunity True Positive Rate between <b>80.1% and 86.8%</b> across all districts, 
            ensuring that domestic violence calls in historically underserved communities are detected with identical high sensitivity to calls in affluent downtown precincts.
            """
        },
        {
            "num": "08",
            "title": "Critical Research Limitations & Non-Predictive Policing Stance",
            "tag": "Scope & Ethics",
            "color": COLORS["rose"],
            "content": """
            <b>CRITICAL ETHICAL CAVEAT:</b> This research is designed strictly for retrospective public safety analytics, resource allocation modeling, and emergency call classification triage. 
            <b>It is NOT intended for predictive policing, individual risk assessment, proactive surveillance, or criminal profiling.</b> 
            Police records reflect reporting patterns, dispatch volume, and systemic administrative practices rather than the true underlying distribution of all human behavior.
            """
        }
    ]
    
    for item in insights:
        st.markdown(f"""
        <div class="panel-card" style="border-left: 4px solid {item['color']}; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h4 style="margin: 0; color: #f8fafc;">{item['num']}. {item['title']}</h4>
                <span class="badge-pill" style="background-color: rgba(255, 255, 255, 0.08); color: {item['color']}; border: 1px solid {item['color']};">{item['tag']}</span>
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
                {item['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
