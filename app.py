import streamlit as st
import pandas as pd

import tempfile


from src.data_preprocessing import load_and_preprocess_data
from src.risk_model import train_risk_model
from src.rag import load_knowledge, build_retriever, retrieve_evidence
from src.pipeline import run_pipeline

# --------------------------------------------------
# Page Configuration - WIDE LAYOUT
# --------------------------------------------------
st.set_page_config(
    page_title="Thyroid Risk Triage – Agentic AI",
    page_icon="🩺",
    layout="wide",  # Full browser width
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Theme Detection & Premium Styling
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ===== CSS Variables for Light/Dark Mode ===== */
:root {
    --bg-primary: #f8fafc;
    --bg-secondary: #ffffff;
    --bg-card: #ffffff;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --accent-primary: #0d4f6e;
    --accent-secondary: #1a7f9e;
    --border-color: #e2e8f0;
    --shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    --shadow-hover: 0 8px 30px rgba(0, 0, 0, 0.12);
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --accent-primary: #38bdf8;
        --accent-secondary: #0ea5e9;
        --border-color: #334155;
        --shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        --shadow-hover: 0 8px 30px rgba(0, 0, 0, 0.4);
    }
}

/* ===== Global Styles ===== */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Full width container override */
.block-container {
    max-width: 1400px !important;
    padding: 2rem 3rem 3rem 3rem !important;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1.5rem !important;
    }
}

/* ===== Header Styling ===== */
.main-header {
    background: linear-gradient(135deg, #0d4f6e 0%, #1a7f9e 50%, #2dd4bf 100%);
    padding: 2.5rem 3rem;
    border-radius: 20px;
    margin-bottom: 2.5rem;
    box-shadow: 0 10px 40px rgba(13, 79, 110, 0.3);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 300px;
    height: 100%;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    opacity: 0.5;
}

.main-header h1 {
    color: #ffffff !important;
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.75rem 0 !important;
    letter-spacing: -0.02em;
}

.main-header p {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
    line-height: 1.6;
    max-width: 700px;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.2);
    color: #ffffff;
    padding: 0.4rem 1rem;
    border-radius: 25px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* ===== Section Cards ===== */
.section-card {
    background: var(--bg-card);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    box-shadow: var(--shadow);
    margin-bottom: 1.75rem;
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.section-card:hover {
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}

.section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.75rem;
    padding-bottom: 1.25rem;
    border-bottom: 2px solid var(--border-color);
}

.section-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #0d4f6e 0%, #1a7f9e 100%);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
}

.section-title {
    color: var(--text-primary) !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

.section-subtitle {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    margin: 0.25rem 0 0 0 !important;
}

/* ===== Form Elements ===== */
.stNumberInput > label,
.stSelectbox > label,
.stCheckbox > label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

.stNumberInput > div > div > input,
/* ===== Selectbox Color Customization ===== */
/* ===== FIX SELECTBOX VISIBILITY (TEXT + ARROW) ===== */

/* Main selectbox container */
div[data-baseweb="select"] > div {
    background-color: #f8fafc !important;   /* light background */
    border: 2px solid #1a7f9e !important;
    border-radius: 12px !important;
    color: #0f172a !important;              /* TEXT COLOR */
}

/* Selected value text */
div[data-baseweb="select"] span {
    color: #0f172a !important;
    font-weight: 500;
}

/* Dropdown arrow (SVG) */
div[data-baseweb="select"] svg {
    fill: #0f172a !important;
}

/* Hover */
div[data-baseweb="select"] > div:hover {
    border-color: #2dd4bf !important;
}

/* Focus (clicked state) */
div[data-baseweb="select"] > div:focus-within {
    background-color: #e0f2fe !important;
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.25) !important;
}

.stCheckbox > label > span {
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
}

.stCheckbox > label > div[data-testid="stCheckbox"] > div {
    background: var(--bg-secondary) !important;
    border-color: var(--border-color) !important;
}

/* ===== Group Labels ===== */
.group-label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.75rem !important;
    display: block;
}

/* ===== Primary Button ===== */
.stButton > button {
    background: linear-gradient(135deg, #0d4f6e 0%, #1a7f9e 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1rem 2.5rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 6px 20px rgba(13, 79, 110, 0.35) !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 30px rgba(13, 79, 110, 0.45) !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* ===== Results Card ===== */
.results-card {
    background: var(--bg-card);
    padding: 2.5rem;
    border-radius: 20px;
    box-shadow: var(--shadow);
    margin-top: 2rem;
    border: 1px solid var(--border-color);
    border-top: 5px solid #1a7f9e;
}

/* ===== Risk Level Badges ===== */
.risk-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.25rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.risk-low {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    color: #065f46;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35);
}

.risk-medium {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    color: #92400e;
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.35);
}

.risk-high {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    color: #991b1b;
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.35);
}

/* ===== Confidence Meter ===== */
.confidence-container {
    background: var(--bg-secondary);
    padding: 1.5rem;
    border-radius: 14px;
    margin: 1.5rem 0;
    border: 1px solid var(--border-color);
}

.confidence-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.75rem;
    font-size: 0.95rem;
    color: var(--text-secondary);
}

.confidence-value {
    color: var(--text-primary);
    font-weight: 700;
    font-size: 1.1rem;
}

.confidence-bar {
    height: 14px;
    background: var(--border-color);
    border-radius: 7px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #1a7f9e 0%, #2dd4bf 100%);
    border-radius: 7px;
    transition: width 0.6s ease;
}

/* ===== Evidence & Precautions ===== */
.evidence-section {
    margin-top: 2rem;
}

.evidence-title {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    margin-bottom: 1rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.evidence-item {
    background: var(--bg-secondary);
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    border-left: 4px solid #1a7f9e;
    font-size: 0.95rem;
    color: var(--text-secondary);
    line-height: 1.7;
    transition: all 0.2s ease;
}

.evidence-item:hover {
    background: var(--bg-primary);
    transform: translateX(4px);
}

.precaution-item {
    background: linear-gradient(135deg, #fef3c7 0%, #fef9c3 100%);
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    border-left: 4px solid #f59e0b;
    font-size: 0.95rem;
    color: #92400e;
    line-height: 1.7;
}

/* Dark mode precaution */
@media (prefers-color-scheme: dark) {
    .precaution-item {
        background: linear-gradient(135deg, #422006 0%, #451a03 100%);
        color: #fcd34d;
    }
}

/* ===== Metrics Override ===== */
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* ===== Alerts ===== */
.stAlert {
    border-radius: 14px !important;
    padding: 1.25rem !important;
}

/* ===== Divider ===== */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%);
    margin: 2rem 0;
}

/* ===== Footer ===== */
.footer {
    text-align: center;
    padding: 2.5rem 1rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border-color);
}

.footer p {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    margin: 0.25rem 0 !important;
}

/* ===== Dark Mode Toggle Indicator ===== */
.theme-indicator {
    position: fixed;
    bottom: 1rem;
    right: 1rem;
    background: var(--bg-card);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.75rem;
    color: var(--text-muted);
    box-shadow: var(--shadow);
    border: 1px solid var(--border-color);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="header-badge">🏥 Clinical Decision Support System</div>
    <h1>🧠 Thyroid Risk Triage – Agentic AI</h1>
    <p>AI-powered risk assessment for thyroid conditions. This system assists clinicians 
    by estimating patient risk levels and providing evidence-based medical insights 
    with supporting precautions.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load System (Cached) - NO CHANGES
# --------------------------------------------------
@st.cache_resource
def load_system():
    df = load_and_preprocess_data("data/Thyroid_Data.csv")

    feature_cols = [
        'age', 'sex',
        'on thyroxine', 'on antithyroid medication', 'sick',
        'pregnant', 'thyroid surgery', 'lithium',
        'goitre', 'tumor', 'hypopituitary', 'psych',
        'TSH', 'T3', 'TT4', 'T4U', 'FTI'
    ]

    model, encoder = train_risk_model(df, feature_cols)

    knowledge = load_knowledge("knowledge/thyroid_knowledge.txt")
    vectorizer, vectors = build_retriever(knowledge)

    def retriever(query):
        return retrieve_evidence(query, vectorizer, vectors, knowledge)

    return model, encoder, retriever, feature_cols


model, encoder, retriever, feature_cols = load_system()

# --------------------------------------------------
# Patient Information Section
# --------------------------------------------------
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-icon">👤</div>
        <div>
            <p class="section-title">Patient Demographics</p>
            <p class="section-subtitle">Enter basic patient information and primary lab values</p>
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=45, help="Patient's current age")

with col2:
    sex = st.selectbox("Biological Sex", ["Male", "Female"], help="Patient's biological sex at birth")

with col3:
    tsh = st.number_input("TSH (mIU/L)", value=2.5, format="%.2f", help="Thyroid Stimulating Hormone")

with col4:
    t3 = st.number_input("T3 (nmol/L)", value=1.2, format="%.2f", help="Triiodothyronine level")

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Thyroid Laboratory Values
# --------------------------------------------------
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-icon">🧪</div>
        <div>
            <p class="section-title">Thyroid Function Panel</p>
            <p class="section-subtitle">Additional laboratory measurements</p>
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    tt4 = st.number_input("TT4 (nmol/L)", value=100.0, format="%.1f", help="Total Thyroxine")

with col2:
    t4u = st.number_input("T4U (ratio)", value=1.0, format="%.2f", help="T4 Uptake ratio")

with col3:
    fti = st.number_input("FTI", value=110.0, format="%.1f", help="Free Thyroxine Index")

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Medical History
# --------------------------------------------------
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-icon">📋</div>
        <div>
            <p class="section-title">Medical History & Current Status</p>
            <p class="section-subtitle">Relevant conditions, medications, and current health status</p>
        </div>
    </div>
""", unsafe_allow_html=True)

is_pregnant_disabled = (sex == "Male")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<span class="group-label">💊 Current Medications</span>', unsafe_allow_html=True)
    on_thyroxine = st.checkbox("On Thyroxine", help="Currently taking thyroxine")
    antithyroid = st.checkbox("On Antithyroid Medication", help="Currently on antithyroid drugs")
    lithium = st.checkbox("On Lithium", help="Currently taking lithium")

with col2:
    st.markdown('<span class="group-label">🏥 Medical Conditions</span>', unsafe_allow_html=True)
    thyroid_surgery = st.checkbox("History of Thyroid Surgery", help="Previous thyroid surgery")
    goitre = st.checkbox("Goitre", help="Enlarged thyroid gland")
    tumor = st.checkbox("Tumor", help="Known thyroid tumor")
    hypopituitary = st.checkbox("Hypopituitary", help="Pituitary dysfunction")

with col3:
    st.markdown('<span class="group-label">📊 Current Status</span>', unsafe_allow_html=True)
    sick = st.checkbox("Currently Sick", help="Patient is currently unwell")
    pregnant = st.checkbox("Pregnant", disabled=is_pregnant_disabled, help="Currently pregnant")
    psych = st.checkbox("Psychological Condition", help="Current psychiatric diagnosis")

if is_pregnant_disabled:
    st.info("ℹ️ Pregnancy option is disabled for male patients.")

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Run Risk Assessment
# --------------------------------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_assessment = st.button("🔬 Run Comprehensive Risk Analysis", use_container_width=True)

if run_assessment:
    # NO CHANGES TO DATA PROCESSING
    new_patient = pd.DataFrame([{
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "on thyroxine": int(on_thyroxine),
        "on antithyroid medication": int(antithyroid),
        "sick": int(sick),
        "pregnant": int(pregnant) if not is_pregnant_disabled else 0,
        "thyroid surgery": int(thyroid_surgery),
        "lithium": int(lithium),
        "goitre": int(goitre),
        "tumor": int(tumor),
        "hypopituitary": int(hypopituitary),
        "psych": int(psych),
        "TSH": tsh,
        "T3": t3,
        "TT4": tt4,
        "T4U": t4u,
        "FTI": fti
    }])

    risk, evidence, precautions, report = run_pipeline(
    model,
    encoder,
    new_patient,
    retriever,
    feature_cols
)


    # --------------------------------------------------
    # Results Display
    # --------------------------------------------------
    st.markdown('<div class="results-card">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <div>
            <p class="section-title">Risk Assessment Report</p>
            <p class="section-subtitle">AI-generated comprehensive risk analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk Level with color coding
    risk_level = risk["risk_level"]
    risk_class = "risk-low" if "low" in risk_level.lower() else ("risk-high" if "high" in risk_level.lower() else "risk-medium")
    risk_icon = "✅" if "low" in risk_level.lower() else ("🔴" if "high" in risk_level.lower() else "⚠️")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem;">
            <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.75rem; font-weight: 500;">ASSESSED RISK LEVEL</p>
            <span class="risk-badge {risk_class}">{risk_icon} {risk_level}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence = risk['confidence']
        confidence_pct = confidence * 100 if confidence <= 1 else confidence
        
        st.markdown(f"""
        <div class="confidence-container">
            <div class="confidence-label">
                <span>Model Confidence Score</span>
                <span class="confidence-value">{confidence_pct:.1f}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence_pct}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Evidence Section
    st.markdown("""
    <div class="evidence-section">
        <p class="evidence-title">📚 Supporting Medical Evidence</p>
    </div>
    """, unsafe_allow_html=True)
    
    for e in evidence:
        st.markdown(f'<div class="evidence-item">{e}</div>', unsafe_allow_html=True)

    # Precautions Section
    st.markdown("""
    <div class="evidence-section">
        <p class="evidence-title">🛡️ General Precautions & Care Guidance</p>
    </div>
    """, unsafe_allow_html=True)
    
    for p in precautions:
        st.markdown(f'<div class="precaution-item">{p}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Warning
    st.warning("⚕️ **Clinical Disclaimer:** This output is for decision support only and must be reviewed by a qualified clinician before any clinical action is taken.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<div class="footer">
    <p><strong>Thyroid Risk Triage System</strong> • Powered by Agentic AI</p>
    <p>For clinical decision support only. Not intended for medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)