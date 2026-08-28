"""
app.py
------
Student Performance AI — Streamlit Web Interface

A polished, modern academic ML application that connects to the
verified Logistic Regression pipeline via src/predict.py.

This file contains ONLY UI code. It does NOT:
- Train or retrain any model
- Modify model hyperparameters
- Perform preprocessing manually
- Fabricate predictions or probabilities

All predictions come directly from the saved pipeline through
predict_performance().

Team Responsibility: Aabiya (UI & Demo Lead)
Core ML Integration: Raghav (src/predict.py)
"""

import streamlit as st
import time

# Import the verified Phase 4 inference function — the ONLY prediction interface
from src.predict import predict_performance

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS — Premium Dark Glassmorphism Theme
# ============================================================
st.markdown("""
<style>
    /* ---- Import Google Fonts ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---- Global Theme ---- */
    .stApp {
        background: linear-gradient(145deg, #0a0e1a 0%, #111827 40%, #0f172a 70%, #0a0e1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ---- Hide Streamlit branding ---- */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---- Glassmorphism Card ---- */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 32px;
        margin: 16px 0;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    /* ---- Hero Section ---- */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
        letter-spacing: -1px;
        line-height: 1.2;
    }

    .hero-subtitle {
        text-align: center;
        color: rgba(148, 163, 184, 0.9);
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 8px;
        letter-spacing: 0.02em;
    }

    .hero-badge {
        text-align: center;
        margin-bottom: 24px;
    }

    .hero-badge span {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* ---- Section Headers ---- */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-header .icon {
        font-size: 1.3rem;
    }

    /* ---- Result Card ---- */
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        padding: 40px 32px;
        text-align: center;
        margin: 20px 0;
        backdrop-filter: blur(20px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: 24px 24px 0 0;
    }

    .result-card.high::before {
        background: linear-gradient(90deg, #22c55e, #10b981, #059669);
    }
    .result-card.average::before {
        background: linear-gradient(90deg, #3b82f6, #6366f1, #8b5cf6);
    }
    .result-card.needs::before {
        background: linear-gradient(90deg, #f59e0b, #ef4444, #dc2626);
    }

    .result-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: rgba(148, 163, 184, 0.8);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 12px;
    }

    .result-category {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 16px;
        line-height: 1.2;
    }

    .result-category.high { color: #4ade80; }
    .result-category.average { color: #818cf8; }
    .result-category.needs { color: #fb923c; }

    .result-probability {
        font-size: 1rem;
        color: rgba(203, 213, 225, 0.9);
        font-weight: 400;
    }

    .result-probability strong {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    /* ---- Probability Bars ---- */
    .prob-container {
        margin: 8px 0;
    }

    .prob-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 14px 0;
        gap: 12px;
    }

    .prob-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #cbd5e1;
        min-width: 160px;
    }

    .prob-bar-bg {
        flex: 1;
        height: 10px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        overflow: hidden;
    }

    .prob-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s ease;
    }

    .prob-bar-fill.high { background: linear-gradient(90deg, #22c55e, #4ade80); }
    .prob-bar-fill.average { background: linear-gradient(90deg, #6366f1, #818cf8); }
    .prob-bar-fill.needs { background: linear-gradient(90deg, #f59e0b, #fb923c); }

    .prob-value {
        font-size: 0.95rem;
        font-weight: 700;
        color: #e2e8f0;
        min-width: 60px;
        text-align: right;
    }

    /* ---- Input Summary ---- */
    .summary-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 12px;
    }

    .summary-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px 20px;
        text-align: center;
    }

    .summary-item .label {
        font-size: 0.78rem;
        font-weight: 600;
        color: rgba(148, 163, 184, 0.7);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }

    .summary-item .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    /* ---- Model Info ---- */
    .model-info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
    }

    .model-info-item {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 14px 18px;
    }

    .model-info-item .mi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: rgba(148, 163, 184, 0.6);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }

    .model-info-item .mi-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #cbd5e1;
    }

    /* ---- Metric Badges ---- */
    .metrics-row {
        display: flex;
        gap: 12px;
        margin-top: 12px;
        flex-wrap: wrap;
    }

    .metric-badge {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 14px;
        padding: 16px 20px;
        flex: 1;
        text-align: center;
        min-width: 120px;
    }

    .metric-badge .mb-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: rgba(148, 163, 184, 0.6);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }

    .metric-badge .mb-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #a5b4fc;
    }

    /* ---- Decorative orb ---- */
    .orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.12;
        pointer-events: none;
        z-index: 0;
    }

    .orb-1 {
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, #6366f1, transparent);
        top: -100px;
        right: -100px;
    }

    .orb-2 {
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, #ec4899, transparent);
        bottom: -50px;
        left: -80px;
    }

    /* ---- Streamlit form button override ---- */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 14px 32px !important;
        border-radius: 14px !important;
        border: none !important;
        width: 100% !important;
        letter-spacing: 0.04em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
    }

    .stFormSubmitButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.45) !important;
    }

    /* ---- Streamlit number input labels ---- */
    .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* ---- Divider ---- */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 24px 0;
        border: none;
    }

    /* ---- Footer ---- */
    .app-footer {
        text-align: center;
        color: rgba(100, 116, 139, 0.6);
        font-size: 0.78rem;
        margin-top: 40px;
        padding-bottom: 20px;
    }
</style>

<!-- Decorative background orbs -->
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
""", unsafe_allow_html=True)


# ============================================================
# HERO SECTION
# ============================================================
st.markdown("""
<div style="margin-top: -20px; margin-bottom: 12px;">
    <div class="hero-title">Student Performance AI</div>
    <div class="hero-subtitle">
        Classical Machine Learning Classification System
    </div>
    <div class="hero-badge">
        <span>LOGISTIC REGRESSION &middot; MULTICLASS</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'input_values' not in st.session_state:
    st.session_state.input_values = None


# ============================================================
# INPUT SECTION
# ============================================================
st.markdown("""
<div class="section-header">
    <span class="icon">📊</span> Enter Academic Indicators
</div>
""", unsafe_allow_html=True)

with st.form("prediction_form", clear_on_submit=False):
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        mst_score = st.number_input(
            "MST Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=0.5,
            help="Mid-Semester Test score (0-100)"
        )
        quiz_score = st.number_input(
            "Quiz Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.5,
            help="Quiz score (0-100)"
        )

    with col2:
        attendance_percent = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.5,
            help="Attendance percentage (0-100)"
        )
        assignment_score = st.number_input(
            "Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=72.0,
            step=0.5,
            help="Assignment score (0-100)"
        )

    st.markdown("")  # Small spacer
    submitted = st.form_submit_button("🔍  ANALYZE PERFORMANCE")


# ============================================================
# PREDICTION LOGIC
# ============================================================
if submitted:
    with st.spinner("Analyzing student performance..."):
        time.sleep(0.6)  # Brief professional loading state (UI only)

        try:
            result = predict_performance(
                mst_score=mst_score,
                quiz_score=quiz_score,
                attendance_percent=attendance_percent,
                assignment_score=assignment_score
            )

            st.session_state.prediction_result = result
            st.session_state.input_values = {
                "MST Score": mst_score,
                "Quiz Score": quiz_score,
                "Attendance": attendance_percent,
                "Assignment Score": assignment_score
            }

        except ValueError as e:
            st.error(f"**Input Validation Error:** {str(e)}")
            st.session_state.prediction_result = None
        except FileNotFoundError as e:
            st.error(f"**Model Not Found:** {str(e)}")
            st.session_state.prediction_result = None
        except Exception as e:
            st.error(f"**Prediction Error:** {str(e)}")
            st.session_state.prediction_result = None


# ============================================================
# DISPLAY RESULTS
# ============================================================
if st.session_state.prediction_result is not None:
    result = st.session_state.prediction_result
    category = result["predicted_category"]
    probabilities = result.get("probabilities", {})
    pred_prob = result.get("predicted_probability", 0)

    # Determine CSS class for color coding
    if category == "High Performer":
        css_class = "high"
    elif category == "Average Performer":
        css_class = "average"
    else:
        css_class = "needs"

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ---- Primary Result Card ----
    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-label">Performance Prediction</div>
        <div class="result-category {css_class}">{category}</div>
        <div class="result-probability">
            Model Probability: <strong>{pred_prob * 100:.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Class Probabilities ----
    st.markdown("""
    <div class="section-header">
        <span class="icon">📈</span> Class Probability Distribution
    </div>
    """, unsafe_allow_html=True)

    prob_high = probabilities.get("High Performer", 0)
    prob_avg = probabilities.get("Average Performer", 0)
    prob_needs = probabilities.get("Needs Improvement", 0)

    st.markdown(f"""
    <div class="glass-card">
        <div class="prob-container">
            <div class="prob-row">
                <span class="prob-label">High Performer</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill high" style="width: {prob_high * 100}%"></div>
                </div>
                <span class="prob-value">{prob_high * 100:.1f}%</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Average Performer</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill average" style="width: {prob_avg * 100}%"></div>
                </div>
                <span class="prob-value">{prob_avg * 100:.1f}%</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Needs Improvement</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill needs" style="width: {prob_needs * 100}%"></div>
                </div>
                <span class="prob-value">{prob_needs * 100:.1f}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Input Summary ----
    if st.session_state.input_values:
        vals = st.session_state.input_values

        st.markdown("""
        <div class="section-header">
            <span class="icon">📋</span> Input Summary
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-card">
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="label">MST Score</div>
                    <div class="value">{vals['MST Score']}</div>
                </div>
                <div class="summary-item">
                    <div class="label">Quiz Score</div>
                    <div class="value">{vals['Quiz Score']}</div>
                </div>
                <div class="summary-item">
                    <div class="label">Attendance</div>
                    <div class="value">{vals['Attendance']}%</div>
                </div>
                <div class="summary-item">
                    <div class="label">Assignment Score</div>
                    <div class="value">{vals['Assignment Score']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Reset Button ----
    st.markdown("")
    if st.button("🔄  Clear & Reset", use_container_width=True):
        st.session_state.prediction_result = None
        st.session_state.input_values = None
        st.rerun()


# ============================================================
# MODEL INFORMATION (Expandable)
# ============================================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

with st.expander("ℹ️  About the Model", expanded=False):
    st.markdown("""
    <div class="model-info-grid">
        <div class="model-info-item">
            <div class="mi-label">Algorithm</div>
            <div class="mi-value">Logistic Regression</div>
        </div>
        <div class="model-info-item">
            <div class="mi-label">Problem Type</div>
            <div class="mi-value">Multiclass Classification</div>
        </div>
        <div class="model-info-item">
            <div class="mi-label">Pipeline</div>
            <div class="mi-value">Imputer &rarr; Scaler &rarr; Classifier</div>
        </div>
        <div class="model-info-item">
            <div class="mi-label">Class Balancing</div>
            <div class="mi-value">class_weight='balanced'</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("**Output Classes:**")
    st.markdown("- 🟢 **High Performer**")
    st.markdown("- 🔵 **Average Performer**")
    st.markdown("- 🟠 **Needs Improvement**")

    st.markdown("")
    st.markdown("**Input Features:**")
    st.markdown("- MST Score (0–100)")
    st.markdown("- Quiz Score (0–100)")
    st.markdown("- Attendance Percentage (0–100)")
    st.markdown("- Assignment Score (0–100)")


# ---- Model Performance (expandable) ----
with st.expander("📊  Model Performance (Held-Out Test Set)", expanded=False):
    st.markdown("""
    <div class="metrics-row">
        <div class="metric-badge">
            <div class="mb-label">Test Accuracy</div>
            <div class="mb-value">94.00%</div>
        </div>
        <div class="metric-badge">
            <div class="mb-label">Macro F1-Score</div>
            <div class="mb-value">91.18%</div>
        </div>
        <div class="metric-badge">
            <div class="mb-label">Weighted F1-Score</div>
            <div class="mb-value">94.21%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.caption(
        "These metrics were evaluated on a held-out 20% test set during Phase 3. "
        "They represent overall model performance, not the probability of any "
        "individual student's prediction."
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="app-footer">
    Student Performance AI &middot; Classical Machine Learning &middot;
    Logistic Regression Pipeline<br>
    Built with Python, Scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)
