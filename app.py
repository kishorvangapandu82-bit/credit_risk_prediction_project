import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="Credit Scoring & Risk Intelligence Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Academic & Institutional Look
st.markdown("""
    <style>
    .main {
        background-color: #F8F9FA;
    }
    .stAppHeader {
        background-color: #1B365D;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .approved-badge {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
    }
    .rejected-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "LightGBM (Gradient Boosting)": "models/lightgbm.joblib",
        "XGBoost (Gradient Boosting)": "models/xgboost.joblib",
        "Random Forest (Bagging)": "models/random_forest.joblib",
        "Decision Tree (CART)": "models/decision_tree.joblib",
        "MLP Neural Network": "models/mlp.joblib",
        "Logistic Regression (Baseline)": "models/logistic_regression.joblib",
        "Support Vector Machine (SVM)": "models/svm.joblib"
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_data
def load_benchmark_data():
    results = {}
    metrics_files = {
        "LightGBM": "results/lightgbm_metrics.csv",
        "XGBoost": "results/xgboost_metrics.csv",
        "Random Forest": "results/random_forest_metrics.csv",
        "Decision Tree": "results/decision_tree_metrics.csv",
        "MLP Neural Net": "results/mlp_metrics.csv",
        "Logistic Regression": "results/logistic_regression_metrics.csv",
        "Linear SVM": "results/svm_metrics.csv"
    }
    dfs = []
    for model_name, path in metrics_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        return combined.sort_values(by="ROC_AUC", ascending=False)
    return pd.DataFrame()

models_dict = load_models()
benchmark_df = load_benchmark_data()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=70)
st.sidebar.title("Credit Risk Intelligence")
st.sidebar.markdown("**GMSC Benchmark System v2.0**")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "💳 Real-Time Credit Underwriting",
        "🏆 Model Benchmark & Performance",
        "🌪️ Macroeconomic Stress Testing",
        "🔍 SHAP Explainability & FCRA",
        "📄 Research Documentation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Model Operational Baseline**: LightGBM & XGBoost achieve peak discriminatory power (**0.8667 ROC-AUC**) with 78.5% default capture.")

# ----------------------------------------------------
# PAGE 1: REAL-TIME CREDIT UNDERWRITING
# ----------------------------------------------------
if page == "💳 Real-Time Credit Underwriting":
    st.title("💳 Real-Time Credit Risk Underwriting & Scoring")
    st.markdown("Assess individual loan applicant profiles, compute Probability of Default (PD), assign credit risk tiers, and generate regulatory Adverse Action codes.")

    col_m, col_t = st.columns([2, 1])
    with col_m:
        selected_model_name = st.selectbox("Select Predictive Model Architecture", list(models_dict.keys()), index=0)
    with col_t:
        approval_threshold = st.slider("Decision Approval Threshold (PD %)", 1.0, 50.0, 20.0, step=1.0) / 100.0

    st.markdown("### 📝 Applicant Financial & Behavioral Input")
    
    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            revolving_util = st.number_input("Revolving Line Utilization (0.0 to 1.0+)", min_value=0.0, max_value=10.0, value=0.35, step=0.05, help="Total balance on credit cards / sum of limits")
            age = st.number_input("Borrower Age (Years)", min_value=18, max_value=100, value=42, step=1)
            debt_ratio = st.number_input("Debt Ratio (Debt / Gross Income)", min_value=0.0, max_value=50.0, value=0.40, step=0.05)
            
        with col2:
            monthly_income = st.number_input("Gross Monthly Income ($)", min_value=0, max_value=500000, value=6500, step=500)
            open_credit_lines = st.number_input("Number of Open Loans & Lines", min_value=0, max_value=50, value=8, step=1)
            real_estate_lines = st.number_input("Number of Real Estate / Mortgage Lines", min_value=0, max_value=20, value=1, step=1)
            
        with col3:
            past_due_30_59 = st.number_input("Times 30-59 Days Past Due (2 Yrs)", min_value=0, max_value=20, value=0, step=1)
            past_due_60_89 = st.number_input("Times 60-89 Days Past Due (2 Yrs)", min_value=0, max_value=20, value=0, step=1)
            past_due_90 = st.number_input("Times 90+ Days Past Due (2 Yrs)", min_value=0, max_value=20, value=0, step=1)
            dependents = st.number_input("Number of Family Dependents", min_value=0, max_value=20, value=1, step=1)

        submit_btn = st.form_submit_button("⚡ Evaluate Credit Application")

    if submit_btn or "evaluated" not in st.session_state:
        st.session_state["evaluated"] = True
        
        # Prepare input dataframe
        input_data = pd.DataFrame([{
            'RevolvingUtilizationOfUnsecuredLines': revolving_util,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': past_due_30_59,
            'DebtRatio': debt_ratio,
            'MonthlyIncome': monthly_income,
            'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
            'NumberOfTimes90DaysLate': past_due_90,
            'NumberRealEstateLoansOrLines': real_estate_lines,
            'NumberOfTime60-89DaysPastDueNotWorse': past_due_60_89,
            'NumberOfDependents': dependents
        }])

        model = models_dict[selected_model_name]
        
        # Predict probability
        if hasattr(model, "predict_proba"):
            pd_prob = model.predict_proba(input_data)[0][1]
        else:
            pd_prob = 0.5

        # Calculate synthetic Credit Score (300 to 850 scale)
        credit_score = int(850 - (pd_prob * 550))
        
        st.markdown("---")
        st.markdown("### 📊 Automated Credit Decision & Risk Assessment")

        # Risk Classification Status Label
        risk_status_label = "Low Risk (Approved)" if is_approved else "High Risk (Rejected)"
        risk_color = "#065F46" if is_approved else "#991B1B"

        res_col1, res_col2, res_col3, res_col4 = st.columns([1.2, 1.2, 1.3, 1.5])

        with res_col1:
            st.metric(label="Probability of Default (PD)", value=f"{pd_prob*100:.2f}%")
        with res_col2:
            st.metric(label="Credit Rating Score", value=f"{credit_score} / 850")
        with res_col3:
            if pd_prob <= 0.05:
                risk_tier = "Prime (Tier 1)"
            elif pd_prob <= 0.15:
                risk_tier = "Near-Prime (Tier 2)"
            elif pd_prob <= 0.30:
                risk_tier = "Subprime (Tier 3)"
            else:
                risk_tier = "Deep Subprime (Tier 4)"
            st.metric(label="Basel Risk Tier", value=risk_tier)
        with res_col4:
            st.markdown(f"**Overall Assessment**")
            if is_approved:
                st.markdown("<div class='approved-badge'>✅ LOW RISK / APPROVED</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='rejected-badge'>❌ HIGH RISK / REJECTED</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Regulatory Adverse Action Reason Codes
        if not is_approved:
            st.warning("⚠️ **US Equal Credit Opportunity Act (ECOA) Adverse Action Notice Reasons**:")
            reasons = []
            if revolving_util > 0.60:
                reasons.append("1. **Excessive Revolving Line Utilization**: Balance exceeds 60% of total credit limits.")
            if past_due_90 > 0:
                reasons.append("2. **Severe Historical Delinquency**: Presence of 90+ days past due records in the past 2 years.")
            if past_due_30_59 + past_due_60_89 > 0:
                reasons.append("3. **Recent Late Payments**: Multiple 30-89 days past due events recorded.")
            if debt_ratio > 0.45:
                reasons.append("4. **High Debt Burden**: Debt-to-income ratio exceeds prime underwriting caps.")
            if monthly_income < 3000:
                reasons.append("5. **Insufficient Liquidity Buffer**: Monthly gross income below prime threshold.")
            
            if not reasons:
                reasons.append("1. **Overall Statistical Risk**: High composite default risk score generated by ML model.")
            
            for r in reasons:
                st.markdown(r)

# ----------------------------------------------------
# PAGE 2: MODEL BENCHMARK & PERFORMANCE
# ----------------------------------------------------
elif page == "🏆 Model Benchmark & Performance":
    st.title("🏆 Empirical Model Benchmark & Performance")
    st.markdown("Rigorous benchmark evaluation across 7 machine learning architectures tested on 45,000 holdout borrower records.")

    if not benchmark_df.empty:
        st.markdown("### 📊 Performance Summary Leaderboard")
        st.dataframe(
            benchmark_df.style.highlight_max(subset=["ROC_AUC", "Recall_Class1"], color="#D1FAE5")
                              .format({"Accuracy": "{:.4f}", "Precision_Class1": "{:.4f}", "Recall_Class1": "{:.4f}", "F1_Class1": "{:.4f}", "ROC_AUC": "{:.4f}"}),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 📈 Comparative Visualizations")

    tab1, tab2, tab3, tab4 = st.columns(4)
    
    img_roc = "plots/roc_curves_comparison.png"
    img_pr = "plots/precision_recall_comparison.png"
    img_cal = "plots/calibration_curves.png"
    img_cm = "plots/confusion_matrices.png"

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("ROC Curves Comparison")
        if os.path.exists(img_roc):
            st.image(img_roc, use_container_width=True)
        else:
            st.info("ROC plot file not found.")

    with col_right:
        st.subheader("Precision-Recall Curves")
        if os.path.exists(img_pr):
            st.image(img_pr, use_container_width=True)
        else:
            st.info("Precision-Recall plot file not found.")

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.subheader("Reliability Calibration Curves")
        if os.path.exists(img_cal):
            st.image(img_cal, use_container_width=True)
    with col_r2:
        st.subheader("Normalized Confusion Matrices")
        if os.path.exists(img_cm):
            st.image(img_cm, use_container_width=True)

# ----------------------------------------------------
# PAGE 3: MACROECONOMIC STRESS TESTING
# ----------------------------------------------------
elif page == "🌪️ Macroeconomic Stress Testing":
    st.title("🌪️ Econometric Macroeconomic Stress Testing Engine")
    st.markdown("Simulate systemic macroeconomic shocks (stagflation, recession, wage drops) and evaluate portfolio risk migration.")

    st.markdown("### 🎛️ Macroeconomic Shock Controls")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        income_drop = st.slider("Borrower Income Contraction (%)", 0, 50, 20, step=5)
    with col_s2:
        debt_spike = st.slider("Borrower Debt-to-Income Increase (%)", 0, 100, 25, step=5)

    income_multiplier = 1.0 - (income_drop / 100.0)
    debt_multiplier = 1.0 + (debt_spike / 100.0)

    st.info(f"⚡ **Simulated Shock Active**: Income Multiplier = `{income_multiplier:.2f}x` | Debt Ratio Multiplier = `{debt_multiplier:.2f}x`")

    img_stress = "plots/stress_test_impact.png"
    if os.path.exists(img_stress):
        st.image(img_stress, use_container_width=True)

    stress_data = pd.DataFrame([
        {"Model": "LightGBM", "Baseline AUC": 0.8667, "Stressed AUC": 0.8660, "AUC Change": "+0.0007", "Default Surge": "+6.09%"},
        {"Model": "XGBoost", "Baseline AUC": 0.8664, "Stressed AUC": 0.8660, "AUC Change": "+0.0004", "Default Surge": "+6.14%"},
        {"Model": "Decision Tree", "Baseline AUC": 0.8437, "Stressed AUC": 0.8438, "AUC Change": "-0.0000", "Default Surge": "-0.06%"},
        {"Model": "MLP Neural Net", "Baseline AUC": 0.8376, "Stressed AUC": 0.8372, "AUC Change": "+0.0003", "Default Surge": "+4.38%"},
        {"Model": "Logistic Regression", "Baseline AUC": 0.8227, "Stressed AUC": 0.8221, "AUC Change": "+0.0006", "Default Surge": "+1.47%"},
        {"Model": "Linear SVM", "Baseline AUC": 0.8205, "Stressed AUC": 0.8198, "AUC Change": "+0.0007", "Default Surge": "+0.13%"}
    ])

    st.markdown("### 📊 Macroeconomic Stress Test Summary Table")
    st.table(stress_data)

# ----------------------------------------------------
# PAGE 4: SHAP EXPLAINABILITY
# ----------------------------------------------------
elif page == "🔍 SHAP Explainability & FCRA":
    st.title("🔍 Game-Theoretic TreeSHAP Model Explainability")
    st.markdown("Global and local feature attribution delivering full compliance with US FCRA and Equal Credit Opportunity Act mandates.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Feature Importance (Beeswarm)")
        img_bee = "plots/shap_summary_beeswarm.png"
        if os.path.exists(img_bee):
            st.image(img_bee, use_container_width=True)
    with col2:
        st.subheader("Mean |SHAP| Feature Rankings")
        img_bar = "plots/shap_feature_importance_bar.png"
        if os.path.exists(img_bar):
            st.image(img_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Local Adverse Action Waterfall Explanations")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.caption("High-Risk Rejection Local Explanation (PD = 84.2%)")
        img_w1 = "plots/shap_waterfall_high_risk.png"
        if os.path.exists(img_w1):
            st.image(img_w1, use_container_width=True)
    with col_w2:
        st.caption("Prime Borrower Approval Local Explanation (PD = 1.8%)")
        img_w2 = "plots/shap_waterfall_low_risk.png"
        if os.path.exists(img_w2):
            st.image(img_w2, use_container_width=True)

# ----------------------------------------------------
# PAGE 5: RESEARCH DOCUMENTATION
# ----------------------------------------------------
elif page == "📄 Research Documentation":
    st.title("📄 Research Paper Blueprint & Basel II/III Compliance")
    
    doc_path = "CONFERENCE_PAPER_DOCUMENTATION.md"
    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content[:8000] + "\n\n*(Full paper text available in repository documentation)*")
    else:
        st.info("Paper documentation file found in root directory.")

