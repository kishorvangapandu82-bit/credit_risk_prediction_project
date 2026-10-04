import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank layout

    # Color Palette Constants
    COLOR_PRIMARY = RGBColor(27, 54, 93)     # Deep Navy Blue
    COLOR_SECONDARY = RGBColor(75, 107, 148)  # Slate Blue
    COLOR_ACCENT = RGBColor(0, 122, 204)     # Vibrant Blue Accent
    COLOR_TEXT_DARK = RGBColor(44, 62, 80)   # Dark Charcoal Body
    COLOR_TEXT_LIGHT = RGBColor(255, 255, 255)# White
    COLOR_BG_LIGHT = RGBColor(248, 249, 250) # Light Off-White
    COLOR_CARD_BG = RGBColor(255, 255, 255)  # Pure White Card
    COLOR_BORDER = RGBColor(220, 224, 230)   # Soft Gray Border
    COLOR_MUTED = RGBColor(100, 110, 120)    # Muted Subtext

    def set_slide_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, category_text, title_text):
        # Category Tracker / Header Banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.9))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.text = category_text.upper()
        p1.font.size = Pt(10)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_ACCENT

        p2 = tf.add_paragraph()
        p2.text = title_text
        p2.font.size = Pt(22)
        p2.font.bold = True
        p2.font.color.rgb = COLOR_PRIMARY

    def add_footer(slide, current_slide, total_slides=14):
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.4))
        tf = footer_box.text_frame
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = f"ML Review 3 Presentation | Credit Default Risk Prediction Benchmark | Slide {current_slide} of {total_slides}"
        p.font.size = Pt(9)
        p.font.color.rgb = COLOR_MUTED

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1)
        else:
            shape.line.fill.background()
        return shape

    # ==========================================
    # SLIDE 1 — PAPER TITLE
    # ==========================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1, COLOR_PRIMARY)

    # Title Card
    tbox = slide1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(5.0))
    tf1 = tbox.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "ML REVIEW 3 PRESENTATION"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    p = tf1.add_paragraph()
    p.text = "An Empirical Benchmark of Machine Learning and Deep Learning Architectures for Credit Default Risk Prediction with Macroeconomic Stress Testing and SHAP Interpretability"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEXT_LIGHT
    p.space_after = Pt(24)

    p = tf1.add_paragraph()
    p.text = "PROJECT METADATA & PRESENTER DETAILS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    p.space_after = Pt(12)

    details = [
        ("Paper / Project Title:", "An Empirical Benchmark of Machine Learning and Deep Learning Architectures for Credit Default Risk Prediction"),
        ("Paper ID:", "[CS-ML-2026-042 / Insert Paper ID]"),
        ("Presenter Name:", "[Sumitranand Sharma / Insert Student Name]"),
        ("Registration Number:", "[22BCE10420 / Insert Registration Number]"),
        ("Target Domain:", "Financial Engineering, Machine Learning, Risk Governance (Basel II/III)")
    ]

    for label, val in details:
        p = tf1.add_paragraph()
        p.text = f"•  {label}  "
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = RGBColor(200, 220, 240)
        run = p.add_run()
        run.text = val
        run.font.bold = False
        run.font.color.rgb = COLOR_TEXT_LIGHT

    # ==========================================
    # SLIDE 2 — PROBLEM STATEMENT AND IDENTIFICATION
    # ==========================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2, COLOR_BG_LIGHT)
    add_header(slide2, "Section 1: Problem Definition", "Problem Statement & Identification")
    add_footer(slide2, 2)

    # 4 Cards Layout
    cards_data = [
        ("1. Problem Being Addressed", "Predicting 2-year borrower credit default probability (SeriousDlqin2yrs) on consumer portfolios to optimize capital reserves and credit risk underwriting under Basel II/III regulatory standards."),
        ("2. Financial Importance", "Capital Reserve Sensitivity: Even a 10–20 bps (0.1–0.2%) inaccuracy in Probability of Default (PD) estimation causes hundreds of millions of dollars in misallocated capital reserves or unmitigated credit losses during recessions."),
        ("3. Current Industry Challenges", "• Extreme Class Imbalance: Default rate is ~6.68% (14:1 ratio).\n• Non-linear Dynamics: Income spikes, age, & debt ratios exhibit non-monotonic risk profiles.\n• Informative Missing Data: ~20% missing income records.\n• Black-Box Opacity: Regulatory mandates (FCRA/ECOA) prohibit unexplainable AI decisions."),
        ("4. Specific Problem Identified", "Traditional scorecard Logistic Regression fails to capture non-linear interaction terms, missing 37.8% of defaults. Published ML literature suffers from data leakage (pre-split imputation/scaling), lacks macroeconomic stress testing, and omits adverse-action XAI compliance.")
    ]

    coords = [(0.8, 1.4), (6.8, 1.4), (0.8, 4.1), (6.8, 4.1)]
    for i, (title, desc) in enumerate(cards_data):
        cx, cy = coords[i]
        add_card(slide2, Inches(cx), Inches(cy), Inches(5.7), Inches(2.5))
        tb = slide2.shapes.add_textbox(Inches(cx+0.15), Inches(cy+0.15), Inches(5.4), Inches(2.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.space_after = Pt(8)
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEXT_DARK

    # ==========================================
    # SLIDE 3 — DATASET USED
    # ==========================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3, COLOR_BG_LIGHT)
    add_header(slide3, "Section 2: Empirical Data", "Dataset Architecture & Characteristics")
    add_footer(slide3, 3)

    # Left Column: Overview Table
    add_card(slide3, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb_l = slide3.shapes.add_textbox(Inches(0.95), Inches(1.5), Inches(5.4), Inches(5.0))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "Dataset Profile: Give Me Some Credit (GMSC)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    ds_info = [
        ("Dataset Source", "Kaggle Benchmark Repository (Public Domain)"),
        ("Total Records (N)", "150,000 Borrower Observations"),
        ("Training Partition", "105,000 Records (70% Stratified Split)"),
        ("Holdout Test Partition", "45,000 Records (30% Untouched Split)"),
        ("Target Variable", "SeriousDlqin2yrs (Binary: 0=Good, 1=Default)"),
        ("Negative Class (0)", "139,974 accounts (93.32% Non-default)"),
        ("Positive Class (1)", "10,026 accounts (6.68% Default)"),
        ("Imbalance Ratio", "13.96 : 1 (Severe Minoritarian Target)")
    ]

    for k, v in ds_info:
        p = tf_l.add_paragraph()
        p.text = f"• {k}: "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_SECONDARY
        r = p.add_run()
        r.text = v
        r.font.bold = False
        r.font.color.rgb = COLOR_TEXT_DARK

    # Right Column: Features & Key Characteristics
    add_card(slide3, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb_r = slide3.shapes.add_textbox(Inches(6.95), Inches(1.5), Inches(5.4), Inches(5.0))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "Predictive Covariates & Quality Anomalies"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    features_text = (
        "10 Baseline Predictive Features:\n"
        "• RevolvingUtilizationOfUnsecuredLines (Ratio)\n"
        "• age (Demographic Integer)\n"
        "• DebtRatio (Financial Burden Ratio)\n"
        "• MonthlyIncome (Gross Currency)\n"
        "• NumberOfOpenCreditLinesAndLoans (Count)\n"
        "• NumberRealEstateLoansOrLines (Count)\n"
        "• Delinquency Recency: 30-59 Days, 60-89 Days, 90+ Days Late\n"
        "• NumberOfDependents (Family Count)\n\n"
        "Data Quality Anomalies Identified:\n"
        "• Informative Missingness: MonthlyIncome (19.82% missing), Dependents (2.62% missing).\n"
        "• Institutional Codes: Delinquency fields contained 96 & 98 (un-contactable codes recoded to NaN).\n"
        "• Biological Anomalies: age=0 recorded in 1 row (recoded to NaN)."
    )

    p2 = tf_r.add_paragraph()
    p2.text = features_text
    p2.font.size = Pt(11)
    p2.font.color.rgb = COLOR_TEXT_DARK

    # ==========================================
    # SLIDE 4 & 5 — LITERATURE REVIEW (6 ARTICLES)
    # ==========================================
    # Slide 4: Literature Review Part 1 (Table view for 6 papers)
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4, COLOR_BG_LIGHT)
    add_header(slide4, "Section 3: Literature Survey", "Literature Review & Comparative Analysis (6 Core Articles)")
    add_footer(slide4, 4)

    # Add a styled table for 6 research papers
    rows, cols = 7, 6
    left, top, width, height = Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3)
    table_shape = slide4.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    col_widths = [Inches(1.8), Inches(0.7), Inches(2.2), Inches(2.0), Inches(2.5), Inches(2.5)]
    for idx, w in enumerate(col_widths):
        table.columns[idx].width = w

    headers = ["Author(s)", "Year", "Paper Title / Focus", "Method / Model", "Key Results & Findings", "Limitations & Project Relevance"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_LIGHT
        p.alignment = PP_ALIGN.CENTER

    lit_data = [
        ("Lessmann et al.", "2015", "Benchmarking Credit Scoring Classifiers", "41 ML algorithms (SVM, RF, NNs) on 8 datasets", "Random Forest & Ensembles outperform Logistic Regression across all datasets.", "Lacked XGBoost/LightGBM evaluation, macroeconomic stress testing, and SHAP XAI."),
        ("Chen & Guestrin", "2016", "XGBoost: Scalable Tree Boosting System", "Gradient Boosted Trees with regularized objectives", "Proved state-of-the-art tabular accuracy & scalable parallel training.", "Lacks domain-specific economic stress testing and regulatory compliance mechanisms."),
        ("Ke et al.", "2017", "LightGBM: Fast Gradient Boosting GDT", "Histogram-based GOSS & Exclusive Feature Bundling", "Achieved 10x faster training speed with equal/higher AUC than XGBoost.", "Evaluated on general benchmarks without cost-asymmetric credit loss functions."),
        ("Lundberg & Lee", "2017", "Unified Interpretability (SHAP)", "Game-theoretic Shapley Additive Explanations", "Mathematically proves unique consistency properties for local explanations.", "Dense networks have high compute overhead; requires TreeSHAP for tree ensembles."),
        ("Grinsztajn et al.", "2022", "Tree Models vs Deep Learning on Tabular", "XGBoost, LightGBM, RF vs MLP on 45 tabular sets", "Gradient boosted trees decisively outperform deep neural nets on tabular data.", "Did not assess credit-specific class imbalance or regulatory adverse action requirements."),
        ("Hand & Henley", "1997", "Statistical Classification in Credit Scoring", "Logistic Regression & Discriminant Analysis", "Established Logistic Regression scorecard as industry benchmark.", "Fails to capture non-linear interaction terms, resulting in poor default recall.")
    ]

    for i, row in enumerate(lit_data):
        for j, val in enumerate(row):
            cell = table.cell(i+1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_BG if i % 2 == 0 else RGBColor(240, 244, 248)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(9)
            p.font.color.rgb = COLOR_TEXT_DARK

    # Slide 5: Literature Summary & Methodological Synthesis
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5, COLOR_BG_LIGHT)
    add_header(slide5, "Section 3: Literature Survey (Contd.)", "Literature Synthesis & Benchmark Methodological Takeaways")
    add_footer(slide5, 5)

    cards_lit_sum = [
        ("Evolution of Credit Scoring Models", "• Phase 1 (1990s): Parametric Linear Models (Hand & Henley, 1997) dominated for regulatory transparency, but suffered high false-negative default losses.\n• Phase 2 (2010s): Ensembles & Boosting (Lessmann 2015, Chen 2016, Ke 2017) demonstrated superior discrimination on complex non-linear financial attributes.\n• Phase 3 (2020s): Deep Tabular vs GBDTs (Grinsztajn 2022) established that tree architectures match un-normalized tabular manifolds far better than deep neural networks."),
        ("Explainability & Regulatory Paradigm Shift", "• The 'Black-Box' Penalty: Financial institutions rejected high-performing ML models due to US FCRA and ECOA mandates requiring explicit Adverse Action reason codes.\n• Game-Theoretic Solution: Lundberg & Lee (2017) introduced SHAP, enabling exact additive feature attributions that bridge the gap between high accuracy GBDTs and regulatory compliance.")
    ]

    cx5 = [0.8, 6.8]
    for i, (title, desc) in enumerate(cards_lit_sum):
        add_card(slide5, Inches(cx5[i]), Inches(1.4), Inches(5.7), Inches(5.3))
        tb = slide5.shapes.add_textbox(Inches(cx5[i]+0.15), Inches(1.5), Inches(5.4), Inches(5.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.space_after = Pt(12)
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEXT_DARK

    # ==========================================
    # SLIDE 6 — RESEARCH GAP ADDRESSED
    # ==========================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6, COLOR_BG_LIGHT)
    add_header(slide6, "Section 4: Research Gap", "Identified Research Gaps & Proposed Project Contributions")
    add_footer(slide6, 6)

    gaps = [
        ("1. Data Leakage in Published Benchmarks", "Existing Literature Strategy: Global dataset imputation and scaling prior to train/test partitioning.\nOur Solution: Strict isolation protocol; imputers and scalers fitted EXCLUSIVELY on X_train (105k) and applied to untouched holdout X_test (45k)."),
        ("2. Omission of Macroeconomic Stress Testing", "Existing Literature Strategy: Static evaluation on clean test splits without testing robustness under recession shocks.\nOur Solution: Econometric stress testing engine simulating a severe stagflation shock (-20% income, +25% debt spike) to verify AUC stability & score migration."),
        ("3. Regulatory Black-Box Barrier", "Existing Literature Strategy: Reporting high accuracy without providing FCRA/ECOA adverse-action justification.\nOur Solution: TreeSHAP game-theoretic explainability generating global feature rankings & local waterfall reason codes for loan approvals and rejections."),
        ("4. Hyperparameter Ceiling Exploration", "Existing Literature Strategy: Ad-hoc manual tuning or computationally prohibitive exhaustive grid searches.\nOur Solution: Automated Bayesian Optimization via Optuna TPE over 50 trials per model family to identify mathematical representation bounds.")
    ]

    c_gaps = [(0.8, 1.4), (6.8, 1.4), (0.8, 4.1), (6.8, 4.1)]
    for i, (title, desc) in enumerate(gaps):
        cx, cy = c_gaps[i]
        add_card(slide6, Inches(cx), Inches(cy), Inches(5.7), Inches(2.5))
        tb = slide6.shapes.add_textbox(Inches(cx+0.15), Inches(cy+0.15), Inches(5.4), Inches(2.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.space_after = Pt(6)
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEXT_DARK

    # ==========================================
    # SLIDE 7 — FEATURE SELECTION & IMPORTANCE ANALYSIS
    # ==========================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7, COLOR_BG_LIGHT)
    add_header(slide7, "Section 5: Feature Engineering", "Feature Selection & Game-Theoretic TreeSHAP Importance")
    add_footer(slide7, 7)

    # Left Column: Text & Rankings
    add_card(slide7, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb7 = slide7.shapes.add_textbox(Inches(0.95), Inches(1.5), Inches(5.4), Inches(5.0))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    p = tf7.paragraphs[0]
    p.text = "TreeSHAP Feature Importance Analysis"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(8)

    feat_desc = (
        "• Selection Methodology: Rather than arbitrary heuristic filtering, we utilized game-theoretic TreeSHAP (SHapley Additive exPlanations) to measure exact marginal contribution across all feature combinations.\n\n"
        "Top Predictive Feature Ranking (Mean |SHAP|):\n"
        "1. RevolvingUtilizationOfUnsecuredLines (0.8247) — Dominant risk factor.\n"
        "2. NumberOfTime30-59DaysPastDueNotWorse (0.3627) — Early delinquency.\n"
        "3. NumberOfTimes90DaysLate (0.3231) — Severe delinquency signal.\n"
        "4. age (0.2322) — Older borrowers exhibit lower risk.\n"
        "5. NumberOfTime60-89DaysPastDueNotWorse (0.1796) — Intermediate signal.\n"
        "6. NumberOfOpenCreditLinesAndLoans (0.1622) — Credit activity.\n"
        "7. DebtRatio (0.1297) — Financial obligation burden.\n"
        "8. MonthlyIncome (0.1113) — Liquidity buffer."
    )
    p2 = tf7.add_paragraph()
    p2.text = feat_desc
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = COLOR_TEXT_DARK

    # Right Column: Embed Image if available
    img_path = "c:/Users/SUMITRANAND SHARMA/OneDrive/Desktop/ML_PRO/credit-scoring-project/plots/shap_summary_beeswarm.png"
    add_card(slide7, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    if os.path.exists(img_path):
        slide7.shapes.add_picture(img_path, Inches(6.9), Inches(1.5), width=Inches(5.5))
    else:
        tb_img = slide7.shapes.add_textbox(Inches(7.0), Inches(3.0), Inches(5.3), Inches(2.0))
        tb_img.text_frame.text = "[SHAP Beeswarm Plot: plots/shap_summary_beeswarm.png]"

    # ==========================================
    # SLIDE 8 — DATASET HANDLING AND DATA PREPROCESSING
    # ==========================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8, COLOR_BG_LIGHT)
    add_header(slide8, "Section 6: Preprocessing", "Data Preprocessing & Anti-Leakage Pipeline")
    add_footer(slide8, 8)

    # 4 Steps Workflow
    prep_steps = [
        ("Step 1: Deterministic Cleaning", "• Biological Invalidity: Set age=0 to NaN.\n• Artifact Recoding: Institutional codes 96 & 98 (un-contactable records) recoded to NaN across all 3 delinquency counts."),
        ("Step 2: Strict Stratified Partition", "• 70% Train (N=105,000) / 30% Holdout Test (N=45,000).\n• Target Default Rate exact match: 6.6838% Train vs 6.6844% Test (Discrepancy < 0.0006%)."),
        ("Step 3: Anti-Leakage Transformation", "• Median Imputation: Fitted ONLY on X_train (Income & Dependents).\n• Outlier Winsorization: Capped at 99th percentile of X_train.\n• Robust Standard Scaling: Fitted exclusively on X_train."),
        ("Step 4: Class Imbalance Handling", "• Class Weighting: Pos-weight scale factor = 13.96 (97,982 / 7,018 ratio).\n• Prevents model bias toward majority non-default class.")
    ]

    c_prep = [(0.8, 1.4), (6.8, 1.4), (0.8, 4.1), (6.8, 4.1)]
    for i, (title, desc) in enumerate(prep_steps):
        cx, cy = c_prep[i]
        add_card(slide8, Inches(cx), Inches(cy), Inches(5.7), Inches(2.5))
        tb = slide8.shapes.add_textbox(Inches(cx+0.15), Inches(cy+0.15), Inches(5.4), Inches(2.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.space_after = Pt(6)
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEXT_DARK

    # ==========================================
    # SLIDE 9 & 10 — ALGORITHMIC DESIGN & APPROACH
    # ==========================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9, COLOR_BG_LIGHT)
    add_header(slide9, "Section 7: Methodology", "Algorithmic Design & Evaluated Model Architectures")
    add_footer(slide9, 9)

    add_card(slide9, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3))
    tb9 = slide9.shapes.add_textbox(Inches(0.95), Inches(1.5), Inches(11.4), Inches(5.0))
    tf9 = tb9.text_frame
    tf9.word_wrap = True

    p = tf9.paragraphs[0]
    p.text = "Taxonomy of 7 Benchmarked Predictive Architectures"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    arch_text = (
        "1. L2-Penalized Logistic Regression: Parametric linear baseline solved via L-BFGS (C=0.001, balanced weights).\n"
        "2. CART Decision Tree: Non-parametric decision tree with Gini impurity criterion (Max depth=6, 59 leaf nodes).\n"
        "3. Support Vector Machine (Linear SVM): Calibrated linear hyperplanes (C=0.01, Platt calibration for probabilities).\n"
        "4. Multi-Layer Perceptron (MLP): Deep feedforward neural network (Architecture: Input(10) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Sigmoid, Adam optimizer, early stopping at epoch 25).\n"
        "5. Random Forest: Bagging ensemble of 200 de-correlated trees (Max depth=12, sqrt feature subsampling).\n"
        "6. Extreme Gradient Boosting (XGBoost): Hist-based gradient boosted decision trees (300 boosting rounds, lr=0.05, max_depth=3, scale_pos_weight=13.96).\n"
        "7. Light Gradient Boosting Machine (LightGBM): Histogram-based GOSS tree boosting (200 rounds, num_leaves=14, lr=0.0453)."
    )
    p2 = tf9.add_paragraph()
    p2.text = arch_text
    p2.font.size = Pt(11)
    p2.font.color.rgb = COLOR_TEXT_DARK

    # Slide 10: Complete Pipeline Flow & Optuna Tuning
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10, COLOR_BG_LIGHT)
    add_header(slide10, "Section 7: Methodology (Contd.)", "End-to-End Pipeline Workflow & Optuna Bayesian Optimization")
    add_footer(slide10, 10)

    add_card(slide10, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb10_l = slide10.shapes.add_textbox(Inches(0.95), Inches(1.5), Inches(5.4), Inches(5.0))
    tf10_l = tb10_l.text_frame
    tf10_l.word_wrap = True

    p = tf10_l.paragraphs[0]
    p.text = "Complete Execution Pipeline Workflow"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    flow_text = (
        "[Raw Borrower Data N=150,000]\n"
        "        │\n"
        "        ▼\n"
        "[Data Cleaning & Code Recoding (96/98 -> NaN)]\n"
        "        │\n"
        "        ▼\n"
        "[Stratified 70/30 Train/Test Partition]\n"
        "        │\n"
        "        ▼\n"
        "[Imputation & Scaling Fitted on X_train ONLY]\n"
        "        │\n"
        "        ▼\n"
        "[Stratified 10-Fold CV & Optuna Bayesian Tuning (50 Trials)]\n"
        "        │\n"
        "        ▼\n"
        "[Model Evaluation on Untouched Holdout X_test (N=45,000)]\n"
        "        │\n"
        "        ▼\n"
        "[Macroeconomic Stress Testing & TreeSHAP Compliance]"
    )
    p2 = tf10_l.add_paragraph()
    p2.text = flow_text
    p2.font.size = Pt(10)
    p2.font.color.rgb = COLOR_TEXT_DARK

    add_card(slide10, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb10_r = slide10.shapes.add_textbox(Inches(6.95), Inches(1.5), Inches(5.4), Inches(5.0))
    tf10_r = tb10_r.text_frame
    tf10_r.word_wrap = True

    p = tf10_r.paragraphs[0]
    p.text = "Optuna Bayesian Optimization Dynamics"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    optuna_text = (
        "• TPE Algorithm: Tree-structured Parzen Estimator evaluated 50 trials per model family across 10-fold CV splits.\n\n"
        "• Selected Best Parameters:\n"
        "  - LightGBM: lr=0.0453, num_leaves=14, n_estimators=200, reg_alpha=2.02, reg_lambda=7.82 (CV-AUC: 0.8654)\n"
        "  - XGBoost: lr=0.0527, max_depth=4, n_estimators=250, reg_alpha=4.11, reg_lambda=8.04 (CV-AUC: 0.8653)\n"
        "  - Random Forest: n_estimators=200, max_depth=12, min_samples_split=50 (CV-AUC: 0.8637)\n"
        "  - MLP: 2 Hidden Layers [64, 32], lr=0.001, alpha=0.001 (CV-AUC: 0.8325)\n\n"
        "• Asymptotic Ceiling Insight: Optimization reached a plateau at ~0.8654 CV-AUC, proving tabular feature representation bounds performance rather than hyperparameter search space."
    )
    p3 = tf10_r.add_paragraph()
    p3.text = optuna_text
    p3.font.size = Pt(10.5)
    p3.font.color.rgb = COLOR_TEXT_DARK

    # ==========================================
    # SLIDE 11 & 12 — RESULTS AND DISCUSSION
    # ==========================================
    # Slide 11: Benchmark Results Table & Analysis
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11, COLOR_BG_LIGHT)
    add_header(slide11, "Section 8: Empirical Results", "Empirical Benchmark Results on Untouched Holdout Test Set (N=45,000)")
    add_footer(slide11, 11)

    # Table of Results
    rows, cols = 8, 8
    left, top, width, height = Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.2)
    t_shape11 = slide11.shapes.add_table(rows, cols, left, top, width, height)
    t11 = t_shape11.table

    w11 = [Inches(0.6), Inches(2.6), Inches(1.3), Inches(1.4), Inches(1.4), Inches(1.3), Inches(1.3), Inches(1.8)]
    for idx, w in enumerate(w11):
        t11.columns[idx].width = w

    headers11 = ["Rank", "Model Architecture", "ROC-AUC", "Recall (Class 1)", "Precision (Class 1)", "F1-Score", "Accuracy", "Test Set Size"]
    for j, h in enumerate(headers11):
        cell = t11.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_LIGHT
        p.alignment = PP_ALIGN.CENTER

    res_data = [
        ("1", "LightGBM (Gradient Boosting)", "0.8667", "0.7852", "0.2126", "0.3346", "0.7912", "45,000"),
        ("2", "XGBoost (Gradient Boosting)", "0.8664", "0.7773", "0.2149", "0.3367", "0.7953", "45,000"),
        ("3", "Random Forest (Bagging)", "0.8639", "0.7512", "0.2179", "0.3377", "0.8015", "45,000"),
        ("4", "Decision Tree (CART)", "0.8437", "0.7354", "0.2251", "0.3447", "0.8131", "45,000"),
        ("5", "MLP Neural Network", "0.8376", "0.7098", "0.2205", "0.3365", "0.8129", "45,000"),
        ("6", "Logistic Regression (Baseline)", "0.8227", "0.6220", "0.2702", "0.3767", "0.8624", "45,000"),
        ("7", "Support Vector Machine (Linear)", "0.8205", "0.1523", "0.5902", "0.2421", "0.9363", "45,000")
    ]

    for i, row in enumerate(res_data):
        for j, val in enumerate(row):
            cell = t11.cell(i+1, j)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = RGBColor(230, 245, 230) # Highlight best
            else:
                cell.fill.fore_color.rgb = COLOR_CARD_BG if i % 2 == 0 else RGBColor(245, 247, 250)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(9.5)
            if j in [1, 2, 3]:
                p.font.bold = True
            p.font.color.rgb = COLOR_PRIMARY if i == 0 else COLOR_TEXT_DARK
            p.alignment = PP_ALIGN.CENTER if j != 1 else PP_ALIGN.LEFT

    # Card below table for key empirical observations
    add_card(slide11, Inches(0.8), Inches(4.8), Inches(11.7), Inches(1.9))
    tb11_b = slide11.shapes.add_textbox(Inches(0.95), Inches(4.9), Inches(11.4), Inches(1.7))
    tf11_b = tb11_b.text_frame
    tf11_b.word_wrap = True

    p = tf11_b.paragraphs[0]
    p.text = "Key Empirical Observations & Financial Interpretation"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(4)

    obs_text = (
        "• Dominant Default Recall: LightGBM (78.52%) and XGBoost (77.73%) catch over 2,362 out of 3,008 default accounts, outperforming Logistic Regression (62.20%) by +16.3 percentage points.\n"
        "• Precision Ceiling (~21.3%): Due to severe 14:1 class imbalance, catching 78% of defaults yields false positives in large negative pools. In banking, False Negative cost (loan loss) is 15-20x higher than False Positive cost (lost margin), making high recall mathematically optimal.\n"
        "• Deep Learning Plateau: MLP (0.8376 AUC) failed to beat Decision Tree ensembles, proving GBDTs handle un-normalized tabular ordinal features far better than neural nets."
    )
    p2 = tf11_b.add_paragraph()
    p2.text = obs_text
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = COLOR_TEXT_DARK

    # Slide 12: Stress Testing & Visual Graphics
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12, COLOR_BG_LIGHT)
    add_header(slide12, "Section 8: Discussion & Stress Testing", "Macroeconomic Stress Testing & Evaluation Plot Visuals")
    add_footer(slide12, 12)

    # Left: Stress Testing Table/Card
    add_card(slide12, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb12_l = slide12.shapes.add_textbox(Inches(0.95), Inches(1.5), Inches(5.4), Inches(5.0))
    tf12_l = tb12_l.text_frame
    tf12_l.word_wrap = True

    p = tf12_l.paragraphs[0]
    p.text = "Macroeconomic Stress Testing Scenario"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(8)

    stress_text = (
        "Simulated Stagflation Shock:\n"
        "• Income Contraction: MonthlyIncome drops by 20% (x 0.80)\n"
        "• Debt Spike: DebtRatio expands by 25% (x 1.25)\n\n"
        "Stress Test Results:\n"
        "• LightGBM: Base AUC 0.8667 -> Stressed 0.8660 (+6.09% default surge)\n"
        "• XGBoost: Base AUC 0.8664 -> Stressed 0.8660 (+6.14% default surge)\n"
        "• MLP: Base AUC 0.8376 -> Stressed 0.8372 (+4.38% default surge)\n"
        "• Logistic Regression: Base AUC 0.8227 -> Stressed 0.8221 (+1.47% default surge)\n\n"
        "Takeaway: Modern GBDTs exhibit >99.9% AUC rank-order stability while dynamically expanding predicted default rates by ~6.1%, allowing financial institutions to raise loss provisions prior to default realization."
    )
    p2 = tf12_l.add_paragraph()
    p2.text = stress_text
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = COLOR_TEXT_DARK

    # Right: ROC Curve & Confusion Matrix Images
    img_roc = "c:/Users/SUMITRANAND SHARMA/OneDrive/Desktop/ML_PRO/credit-scoring-project/plots/roc_curves_comparison.png"
    add_card(slide12, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    if os.path.exists(img_roc):
        slide12.shapes.add_picture(img_roc, Inches(6.9), Inches(1.5), width=Inches(5.5))
    else:
        tb_img2 = slide12.shapes.add_textbox(Inches(7.0), Inches(3.0), Inches(5.3), Inches(2.0))
        tb_img2.text_frame.text = "[ROC Curves Comparison: plots/roc_curves_comparison.png]"

    # ==========================================
    # SLIDE 13 — CONCLUSION
    # ==========================================
    slide13 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide13, COLOR_BG_LIGHT)
    add_header(slide13, "Section 9: Conclusion", "Conclusion, Research Impact & Future Directions")
    add_footer(slide13, 13)

    concl_cards = [
        ("1. Key Achievements", "• Built a data-leakage-free, regulatory-compliant benchmark comparing 7 ML/DL architectures on 150,000 borrower profiles.\n• LightGBM & XGBoost achieved supreme performance (0.8667 & 0.8664 ROC-AUC, ~78.5% default recall)."),
        ("2. Research Gaps Addressed", "• Preprocessing isolation eliminated data leakage bias.\n• Econometric stress testing proved GBDT rank-order stability under stagflation shocks.\n• TreeSHAP provided FCRA/ECOA regulatory adverse action compliance."),
        ("3. Practical Significance", "• Transitioning from scorecard Logistic Regression to LightGBM captures +16.3% more defaulting accounts, saving banks millions in credit write-offs.\n• Dynamic default expansion (+6.1%) enables proactive capital provisioning."),
        ("4. Future Scope", "• Multi-year macroeconomic survival analysis for Loss Given Default (LGD) time-to-event modeling.\n• Automated feature synthesis & self-supervised tabular pre-training.")
    ]

    c_concl = [(0.8, 1.4), (6.8, 1.4), (0.8, 4.1), (6.8, 4.1)]
    for i, (title, desc) in enumerate(concl_cards):
        cx, cy = c_concl[i]
        add_card(slide13, Inches(cx), Inches(cy), Inches(5.7), Inches(2.5))
        tb = slide13.shapes.add_textbox(Inches(cx+0.15), Inches(cy+0.15), Inches(5.4), Inches(2.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.space_after = Pt(6)
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEXT_DARK

    # ==========================================
    # SLIDE 14 — THANK YOU
    # ==========================================
    slide14 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide14, COLOR_PRIMARY)

    tb14 = slide14.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.333), Inches(4.0))
    tf14 = tb14.text_frame
    tf14.word_wrap = True

    p = tf14.paragraphs[0]
    p.text = "THANK YOU!"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(20)

    p2 = tf14.add_paragraph()
    p2.text = "Questions & Academic Discussion"
    p2.font.size = Pt(20)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_TEXT_LIGHT
    p2.alignment = PP_ALIGN.CENTER
    p2.space_after = Pt(30)

    p3 = tf14.add_paragraph()
    p3.text = "Presenter: [Sumitranand Sharma / Insert Student Name]\nRegistration Number: [22BCE10420 / Insert Registration Number]\nPaper Title: An Empirical Benchmark of Machine Learning and Deep Learning Architectures for Credit Default Risk Prediction"
    p3.font.size = Pt(14)
    p3.font.color.rgb = RGBColor(200, 220, 240)
    p3.alignment = PP_ALIGN.CENTER

    output_path = "c:/Users/SUMITRANAND SHARMA/OneDrive/Desktop/ML_PRO/credit-scoring-project/ML_Review_3_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")

if __name__ == "__main__":
    create_presentation()
