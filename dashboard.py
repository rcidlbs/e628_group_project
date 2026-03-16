"""
Lending Club — Credit Risk & Loan Default Prediction
Dash Dashboard  |  dashboard.py

Run:
    python dashboard.py
Then open:  http://127.0.0.1:8050

Requirements:
    pip install dash dash-bootstrap-components plotly shap xgboost scikit-learn pandas numpy matplotlib
"""

# ─────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────
import io, base64, warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score
)

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these values to change pipeline behaviour
# ─────────────────────────────────────────────────────────────────
CSV_FILE     = "lending_club_sample.csv"  # cleaned file produced by EDA notebook
NROWS        = 50_000  # Reduced for Render Free (512 MB RAM). Set to None to use all rows locally.
TEST_SIZE    = 0.2     # fraction held out for evaluation (try 0.1 or 0.3)
RANDOM_STATE = 42      # controls train/test split and all model random seeds

# ─────────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────
print(f"Loading data from '{CSV_FILE}' (nrows={NROWS}, test_size={TEST_SIZE})...")
df_raw = pd.read_csv(CSV_FILE, nrows=NROWS, low_memory=False)

for col in ["revol_util", "bc_util", "pct_tl_nvr_dlq", "percent_bc_gt_75"]:
    if col in df_raw.columns:
        df_raw[col] = df_raw[col].fillna(df_raw[col].median())
for col in ["mort_acc", "pub_rec_bankruptcies", "num_accts_ever_120_pd"]:
    if col in df_raw.columns:
        df_raw[col] = df_raw[col].fillna(0)
if "emp_length" in df_raw.columns:
    df_raw["emp_length"] = df_raw["emp_length"].fillna(df_raw["emp_length"].mode()[0])

# Parse issue_d if present (needed for time-series analysis)
# infer_datetime_format handles "Jan-2015", "2015-01-01", "01/2015", etc.
if "issue_d" in df_raw.columns:
    df_raw["issue_d"] = pd.to_datetime(df_raw["issue_d"], errors="coerce")

default_statuses = ["Charged Off","Late (16-30 days)","Late (31-120 days)","In Grace Period","Default"]
df_raw["default"]       = df_raw["loan_status"].isin(default_statuses).astype(int)
df_raw["default_label"] = df_raw["default"].map({0: "No Default", 1: "Default"})

if "grade_num" not in df_raw.columns:
    df_raw["grade_num"] = df_raw["grade"].map({"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7})
if "term_months" not in df_raw.columns:
    df_raw["term_months"] = df_raw["term"].str.replace(" months","").str.strip().astype(float)
if "emp_length_yrs" not in df_raw.columns:
    el = df_raw["emp_length"].astype(str)
    el = el.str.replace("< 1 year","0").str.replace("10+ years","10")
    el = el.str.replace(" years","").str.replace(" year","")
    df_raw["emp_length_yrs"] = pd.to_numeric(el, errors="coerce").fillna(0)
if "fico_avg" not in df_raw.columns:
    df_raw["fico_avg"] = (df_raw["fico_range_low"] + df_raw["fico_range_high"]) / 2

df_raw["installment_to_income"] = df_raw["installment"] / (df_raw["annual_inc"].clip(lower=1) / 12)
if "earliest_cr_line" in df_raw.columns:
    df_raw["earliest_cr_line"] = pd.to_datetime(df_raw["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df_raw["credit_history_years"] = (pd.Timestamp("2018-12-31") - df_raw["earliest_cr_line"]).dt.days / 365.25
else:
    df_raw["credit_history_years"] = 10.0
df_raw["delinquency_flag"] = (
    (df_raw.get("delinq_2yrs", 0) > 0) | (df_raw.get("num_accts_ever_120_pd", 0) > 0)
).astype(int)

df = df_raw.copy()
print(f"  Rows: {len(df):,}  |  Default rate: {df['default'].mean():.1%}")

# ─────────────────────────────────────────────────────────────────
# 2. MODELING
# ─────────────────────────────────────────────────────────────────
print("Running feature engineering for modeling...")
df_model = df.copy()
df_model = pd.get_dummies(
    df_model,
    columns=["home_ownership","purpose","verification_status","application_type"],
    drop_first=True
)
cols_to_drop = ["loan_status","default_label","grade","sub_grade","term","emp_length",
                "addr_state","earliest_cr_line","fico_range_low","fico_range_high"]
cols_to_drop = [c for c in cols_to_drop if c in df_model.columns]

y = df_model["default"]
X = df_model.drop(columns=cols_to_drop + ["default"])
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# ─────────────────────────────────────────────────────────────────
# LOAD OR TRAIN MODELS
# If models/ folder exists, load pre-trained models (fast: ~15 sec).
# Otherwise train from scratch (slow: ~4 min).
# Run save_models.ipynb once locally to generate the models/ folder.
# ─────────────────────────────────────────────────────────────────
import joblib

MODELS_DIR = "models"

if os.path.exists(os.path.join(MODELS_DIR, "xgb.pkl")):
    print("Loading pre-trained models from models/ ...")
    lr     = joblib.load(f"{MODELS_DIR}/lr.pkl")
    dt     = joblib.load(f"{MODELS_DIR}/dt.pkl")
    rf     = joblib.load(f"{MODELS_DIR}/rf.pkl")
    xgb    = joblib.load(f"{MODELS_DIR}/xgb.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")

    eval_data         = joblib.load(f"{MODELS_DIR}/eval_data.pkl")
    X_test            = eval_data["X_test"]
    y_test            = eval_data["y_test"]
    proba_lr          = eval_data["proba_lr"]
    proba_dt          = eval_data["proba_dt"]
    proba_rf          = eval_data["proba_rf"]
    proba_xgb         = eval_data["proba_xgb"]
    OPTIMAL_THRESHOLD = eval_data["optimal_threshold"]
    thresholds        = eval_data["thresholds"]
    f1_scores         = eval_data["f1_scores"]
    TRAIN_MEDIANS     = eval_data["X_train_medians"]
    FEATURE_COLS      = eval_data["feature_cols"]
    print(f"  Models loaded  |  Optimal threshold: {OPTIMAL_THRESHOLD:.2f}")

else:
    print("models/ not found — training from scratch (this may take ~4 minutes)...")
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_sc, y_train)
    print("  Logistic Regression ✓")

    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50,
         class_weight="balanced", random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    print("  Decision Tree ✓")

    rf = RandomForestClassifier(n_estimators=50, max_features="sqrt", max_depth=15,
         min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("  Random Forest ✓")

    neg, pos = (y_train==0).sum(), (y_train==1).sum()
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=round(neg/pos,2),
        eval_metric="logloss", random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    print("  XGBoost ✓")

    proba_lr  = lr.predict_proba(X_test_sc)[:,1]
    proba_dt  = dt.predict_proba(X_test)[:,1]
    proba_rf  = rf.predict_proba(X_test)[:,1]
    proba_xgb = xgb.predict_proba(X_test)[:,1]

    thresholds        = np.arange(0.05, 0.95, 0.01)
    f1_scores         = [f1_score(y_test, (proba_xgb >= t).astype(int), pos_label=1) for t in thresholds]
    OPTIMAL_THRESHOLD = round(float(thresholds[np.argmax(f1_scores)]), 2)
    TRAIN_MEDIANS     = X_train.median().to_dict()
    FEATURE_COLS      = list(X.columns)
    print(f"  Optimal threshold (max F1): {OPTIMAL_THRESHOLD:.2f}")

# ─────────────────────────────────────────────────────────────────
# 3. SHAP — load pre-generated PNGs or generate on first run
# ─────────────────────────────────────────────────────────────────

def _fig_to_b64():
    """Convert current matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=72)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close("all")
    return f"data:image/png;base64,{b64}"


def _png_to_b64(path):
    """Load a PNG file from disk and return as base64 string."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


SHAP_PNG_FILES = {
    "beeswarm":       f"{MODELS_DIR}/shap_beeswarm.png",
    "bar":            f"{MODELS_DIR}/shap_bar.png",
    "waterfall_high": f"{MODELS_DIR}/shap_waterfall_high.png",
    "waterfall_low":  f"{MODELS_DIR}/shap_waterfall_low.png",
}

if all(os.path.exists(p) for p in SHAP_PNG_FILES.values()):
    print("Loading pre-generated SHAP plots from models/ ...")
    SHAP_BEESWARM      = _png_to_b64(SHAP_PNG_FILES["beeswarm"])
    SHAP_BAR           = _png_to_b64(SHAP_PNG_FILES["bar"])
    SHAP_WATERFALL_HIGH = _png_to_b64(SHAP_PNG_FILES["waterfall_high"])
    SHAP_WATERFALL_LOW  = _png_to_b64(SHAP_PNG_FILES["waterfall_low"])
    print("  SHAP plots loaded ✓")
else:
    print("SHAP PNGs not found — generating now (this takes ~1 minute)...")
    np.random.seed(42)
    shap_idx    = np.random.choice(len(X_test), size=500, replace=False)
    X_shap      = X_test.iloc[shap_idx].reset_index(drop=True)
    explainer   = shap.TreeExplainer(xgb.get_booster())
    shap_values = explainer(X_shap)

    proba_shap    = xgb.predict_proba(X_shap)[:,1]
    high_risk_idx = int(np.argsort(proba_shap)[-1])
    low_risk_idx  = int(np.argsort(proba_shap)[0])

    plt.figure(figsize=(10,7))
    shap.summary_plot(shap_values.values, X_shap, max_display=15, show=False)
    plt.title("SHAP Summary — Global Feature Impact & Direction", fontweight="bold")
    plt.tight_layout()
    SHAP_BEESWARM = _fig_to_b64()

    plt.figure(figsize=(9,6))
    shap.summary_plot(shap_values.values, X_shap, plot_type="bar", max_display=15, show=False)
    plt.title("SHAP Feature Importance (mean |SHAP value|)", fontweight="bold")
    plt.tight_layout()
    SHAP_BAR = _fig_to_b64()

    plt.figure(figsize=(10,6))
    shap.plots.waterfall(shap_values[high_risk_idx], max_display=12, show=False)
    plt.title(f"High-Risk Borrower  (p = {proba_shap[high_risk_idx]:.1%})", fontweight="bold")
    plt.tight_layout()
    SHAP_WATERFALL_HIGH = _fig_to_b64()

    plt.figure(figsize=(10,6))
    shap.plots.waterfall(shap_values[low_risk_idx], max_display=12, show=False)
    plt.title(f"Low-Risk Borrower  (p = {proba_shap[low_risk_idx]:.1%})", fontweight="bold")
    plt.tight_layout()
    SHAP_WATERFALL_LOW = _fig_to_b64()
    print("  SHAP done ✓")

# ─────────────────────────────────────────────────────────────────
# 3b. GLOBAL SHAP EXPLAINER (created once, reused in Score Calculator)
# ─────────────────────────────────────────────────────────────────
print("Initializing SHAP explainer...")
SHAP_EXPLAINER = shap.TreeExplainer(xgb.get_booster())
print("  SHAP explainer ready ✓")

# ─────────────────────────────────────────────────────────────────
# 4. STATIC PLOTLY FIGURES
# ─────────────────────────────────────────────────────────────────
C = {"green":"#1a9e6b","red":"#d63031","blue":"#0984e3",
     "orange":"#e17055","grey":"#636e72","bg":"#f0f3f7","dark":"#2d3436"}
MODEL_COLORS = ["#3498db","#e67e22","#27ae60","#e74c3c"]
MODEL_NAMES  = ["Logistic Regression","Decision Tree","Random Forest","XGBoost"]
AVG_DEFAULT  = df["default"].mean()

status_counts = df["loan_status"].value_counts().reset_index()
status_counts.columns = ["loan_status","count"]
fig_status = px.bar(status_counts, y="loan_status", x="count", orientation="h",
    color_discrete_sequence=["#2980b9"],
    labels={"count":"Number of loans","loan_status":""},
    title="Distribution of Loan Status")
fig_status.update_layout(plot_bgcolor=C["bg"])

default_counts = df["default_label"].value_counts().reset_index()
default_counts.columns = ["label","count"]
fig_pie = px.pie(default_counts, names="label", values="count",
    color="label", color_discrete_map={"No Default":C["green"],"Default":C["red"]},
    title="Binary Target: Default vs No Default", hole=0.4)

grade_dr = df.groupby("grade")["default"].mean().reset_index()
grade_dr.columns = ["grade","default_rate"]
grade_dr = grade_dr.sort_values("grade")
fig_grade = px.bar(grade_dr, x="grade", y="default_rate",
    color="default_rate", color_continuous_scale="RdYlGn_r",
    labels={"default_rate":"Default Rate","grade":"Grade"},
    title="Default Rate by Grade",
    text=grade_dr["default_rate"].apply(lambda v: f"{v:.1%}"))
fig_grade.add_hline(y=AVG_DEFAULT, line_dash="dash", line_color="black",
    annotation_text=f"Avg: {AVG_DEFAULT:.1%}")
fig_grade.update_traces(textposition="outside")
fig_grade.update_layout(coloraxis_showscale=False, plot_bgcolor=C["bg"])

purpose_dr = df.groupby("purpose")["default"].mean().reset_index()
purpose_dr.columns = ["purpose","default_rate"]
purpose_dr = purpose_dr.sort_values("default_rate")
fig_purpose = px.bar(purpose_dr, y="purpose", x="default_rate", orientation="h",
    color="default_rate", color_continuous_scale="RdYlGn_r",
    labels={"default_rate":"Default Rate","purpose":""},
    title="Default Rate by Loan Purpose",
    text=purpose_dr["default_rate"].apply(lambda v: f"{v:.1%}"))
fig_purpose.add_vline(x=AVG_DEFAULT, line_dash="dash", line_color="black",
    annotation_text=f"Avg: {AVG_DEFAULT:.1%}")
fig_purpose.update_traces(textposition="outside")
fig_purpose.update_layout(coloraxis_showscale=False, plot_bgcolor=C["bg"], height=500)

home_dr = (df[df["home_ownership"].isin(["OWN","MORTGAGE","RENT"])]
    .groupby("home_ownership")["default"].mean().reset_index())
home_dr.columns = ["home_ownership","default_rate"]
home_dr = home_dr.sort_values("default_rate")
fig_home = px.bar(home_dr, x="home_ownership", y="default_rate",
    color="home_ownership",
    color_discrete_sequence=[C["green"],C["orange"],C["red"]],
    labels={"default_rate":"Default Rate","home_ownership":""},
    title="Default Rate by Home Ownership",
    text=home_dr["default_rate"].apply(lambda v: f"{v:.1%}"))
fig_home.add_hline(y=AVG_DEFAULT, line_dash="dash", line_color="black",
    annotation_text=f"Avg: {AVG_DEFAULT:.1%}")
fig_home.update_traces(textposition="outside")
fig_home.update_layout(showlegend=False, plot_bgcolor=C["bg"])

verif_dr = df.groupby("verification_status")["default"].mean().reset_index()
verif_dr.columns = ["verification_status","default_rate"]
verif_dr = verif_dr.sort_values("default_rate")
fig_verif = px.bar(verif_dr, x="verification_status", y="default_rate",
    color="verification_status",
    color_discrete_sequence=[C["green"],C["orange"],C["red"]],
    labels={"default_rate":"Default Rate","verification_status":""},
    title="Default Rate by Verification Status",
    text=verif_dr["default_rate"].apply(lambda v: f"{v:.1%}"))
fig_verif.add_hline(y=AVG_DEFAULT, line_dash="dash", line_color="black",
    annotation_text=f"Avg: {AVG_DEFAULT:.1%}")
fig_verif.update_traces(textposition="outside")
fig_verif.update_layout(showlegend=False, plot_bgcolor=C["bg"])

cols_corr = ["fico_avg","int_rate","dti","annual_inc","revol_util","loan_amnt",
             "delinq_2yrs","inq_last_6mths","open_acc","pub_rec","total_acc","grade_num","default"]
corr_matrix = df[[c for c in cols_corr if c in df.columns]].corr().round(2)
fig_heatmap = px.imshow(corr_matrix, color_continuous_scale="RdYlGn",
    zmin=-1, zmax=1, text_auto=True, title="Correlation Heatmap", aspect="auto")
fig_heatmap.update_layout(height=550)

corr_target = corr_matrix["default"].drop("default").sort_values()
fig_corr_bar = go.Figure(go.Bar(
    x=corr_target.values, y=corr_target.index, orientation="h",
    marker_color=[C["red"] if v>0 else C["green"] for v in corr_target.values],
    text=[f"{v:.2f}" for v in corr_target.values], textposition="outside"))
fig_corr_bar.add_vline(x=0, line_color="black", line_width=1)
fig_corr_bar.update_layout(
    title="Correlation with Default",
    xaxis_title="Pearson Correlation", plot_bgcolor=C["bg"], height=450)

fig_roc = go.Figure()
for (name, proba), color in zip(
    [("Logistic Regression",proba_lr),("Decision Tree",proba_dt),
     ("Random Forest",proba_rf),("XGBoost",proba_xgb)], MODEL_COLORS):
    auc = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
        name=f"{name} (AUC={auc:.3f})", line=dict(color=color, width=2)))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random (0.500)",
    line=dict(color="grey", dash="dash", width=1)))
fig_roc.update_layout(title="ROC Curve — All Models",
    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    plot_bgcolor=C["bg"], hovermode="x unified")

fig_pr = go.Figure()
for (name, proba), color in zip(
    [("Logistic Regression",proba_lr),("Decision Tree",proba_dt),
     ("Random Forest",proba_rf),("XGBoost",proba_xgb)], MODEL_COLORS):
    ap = average_precision_score(y_test, proba)
    precision, recall, _ = precision_recall_curve(y_test, proba)
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
        name=f"{name} (AP={ap:.3f})", line=dict(color=color, width=2)))
fig_pr.add_hline(y=y_test.mean(), line_dash="dash", line_color="grey",
    annotation_text=f"Random ({y_test.mean():.2f})")
fig_pr.update_layout(title="Precision-Recall Curve — All Models",
    xaxis_title="Recall (defaults caught)",
    yaxis_title="Precision (flagged loans that truly default)", plot_bgcolor=C["bg"])

fig_f1 = go.Figure()
fig_f1.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode="lines",
    line=dict(color="#9b59b6", width=2), name="F1 Score"))
fig_f1.add_vline(x=OPTIMAL_THRESHOLD, line_dash="dash", line_color=C["red"],
    annotation_text=f"Optimal = {OPTIMAL_THRESHOLD:.2f}", annotation_position="top right")
fig_f1.update_layout(
    title=f"F1 Score vs Threshold (XGBoost) — Optimal = {OPTIMAL_THRESHOLD:.2f}",
    xaxis_title="Decision Threshold", yaxis_title="F1 Score", plot_bgcolor=C["bg"])


def _fi_fig(model, title):
    imp = pd.Series(model.feature_importances_, index=X.columns)
    top = imp.sort_values().tail(15)
    f = px.bar(x=top.values, y=top.index, orientation="h",
        color_discrete_sequence=["steelblue"],
        labels={"x":"Importance","y":""}, title=title)
    f.update_layout(plot_bgcolor=C["bg"], height=500)
    return f


fig_fi_rf  = _fi_fig(rf,  "Random Forest — Top 15 Feature Importances")
fig_fi_xgb = _fi_fig(xgb, "XGBoost — Top 15 Feature Importances")

decile_df2 = pd.DataFrame({"actual":y_test.values,"proba":proba_xgb})
decile_df2["decile"] = pd.qcut(decile_df2["proba"], q=10, labels=range(1,11))
decile_sum = decile_df2.groupby("decile").agg(
    count=("actual","count"), actual_default_rate=("actual","mean")).reset_index()
fig_decile = px.bar(decile_sum, x="decile", y="actual_default_rate",
    color="actual_default_rate", color_continuous_scale="RdYlGn_r",
    labels={"actual_default_rate":"Actual Default Rate","decile":"Risk Decile"},
    title="Actual Default Rate by Predicted Risk Decile (XGBoost)",
    text=decile_sum["actual_default_rate"].apply(lambda v: f"{v:.1%}"),
    hover_data=["count"])
fig_decile.add_hline(y=y_test.mean(), line_dash="dash", line_color="black",
    annotation_text=f"Overall avg: {y_test.mean():.1%}")
fig_decile.update_traces(textposition="outside")
fig_decile.update_layout(coloraxis_showscale=False, plot_bgcolor=C["bg"])

results = []
for name, proba, thresh in zip(
        MODEL_NAMES,
        [proba_lr,proba_dt,proba_rf,proba_xgb],
        [0.5,0.5,0.5,OPTIMAL_THRESHOLD]):
    auc  = roc_auc_score(y_test, proba)
    gini = 2*auc - 1
    ap   = average_precision_score(y_test, proba)
    y_pred = (proba >= thresh).astype(int)
    rep  = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        "Model":     name,
        "AUC-ROC":   round(auc,  4),
        "Gini":      round(gini, 4),
        "Avg Prec":  round(ap,   4),
        "Precision": round(rep["1"]["precision"], 4),
        "Recall":    round(rep["1"]["recall"],    4),
        "F1":        round(rep["1"]["f1-score"],  4),
    })
results_df = pd.DataFrame(results).sort_values("AUC-ROC")

# Build HTML summary table
_metrics        = ["AUC-ROC","Gini","Avg Prec","Precision","Recall","F1"]
_best           = {m: results_df[m].max() for m in _metrics}
_sorted_results = results_df.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)

def _metric_cell(val, best_val):
    is_best = (abs(val - best_val) < 1e-9)
    cell_style = {"fontWeight":"700","color":"#1a9e6b"} if is_best else {"color":"#2d3436"}
    return html.Td(f"{val:.4f}", style=cell_style)

def _build_summary_table():
    rows = []
    for _, row in _sorted_results.iterrows():
        model_name = str(row["Model"])
        is_xgb     = (model_name == "XGBoost")
        row_style  = {"background":"#eaf6f0","fontWeight":"600"} if is_xgb else {}
        name_cell  = html.Td(html.Strong(model_name) if is_xgb else model_name)
        metric_cells = [_metric_cell(float(row[m]), _best[m]) for m in _metrics]
        rows.append(html.Tr([name_cell] + metric_cells, style=row_style))
    return dbc.Table([
        html.Thead(
            html.Tr(
                [html.Th("Model", style={"minWidth":"160px","color":"#2d3436","borderBottom":"2px solid #0984e3"})] +
                [html.Th(m, style={"color":"#2d3436","borderBottom":"2px solid #0984e3","fontWeight":"700"})
                 for m in _metrics]),
            style={"background":"#eaf0fb","fontSize":"0.83rem"}),
        html.Tbody(rows),
    ], bordered=False, hover=True, responsive=True, size="sm",
       style={"fontSize":"0.88rem","marginBottom":"8px",
              "borderCollapse":"collapse","border":"1px solid #dfe6e9"})

summary_table = _build_summary_table()
fig_summary = px.bar(results_df, y="Model", x="AUC-ROC", orientation="h",
    color="AUC-ROC", color_continuous_scale="Blues",
    text=results_df["AUC-ROC"].apply(lambda v: f"{v:.3f}"),
    title="Model Comparison — AUC-ROC")
fig_summary.add_vline(x=0.5, line_dash="dash", line_color="red",
    annotation_text="Random = 0.5")
fig_summary.update_traces(textposition="outside")
fig_summary.update_layout(coloraxis_showscale=False, plot_bgcolor=C["bg"])

# ─────────────────────────────────────────────────────────────────
# 5. UI HELPERS
# ─────────────────────────────────────────────────────────────────
def insight_box(text, color="#0984e3"):
    return html.Div(
        html.P([html.Strong("Key finding: "), text], className="mb-0"),
        style={
            "background": f"{color}12",
            "borderLeft": f"4px solid {color}",
            "padding": "11px 16px",
            "borderRadius": "0 6px 6px 0",
            "marginBottom": "18px",
            "marginTop": "6px",
            "fontSize": "0.91rem",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
        })


def concept_box(title, text):
    return html.Div([
        html.Div([
            html.Span(title, style={
                "fontWeight": "700",
                "fontSize": "0.82rem",
                "textTransform": "uppercase",
                "letterSpacing": "0.06em",
                "color": "#636e72",
            }),
        ], style={"marginBottom": "4px"}),
        html.P(text, className="mb-0",
               style={"fontSize": "0.89rem", "color": "#2d3436", "lineHeight": "1.55"}),
    ], style={
        "background": "#eaf0fb",
        "border": "1px solid #c8d6f5",
        "borderRadius": "6px",
        "padding": "12px 16px",
        "marginBottom": "14px",
    })


def section_header(title, subtitle=None):
    return html.Div([
        html.H5(title, style={"fontWeight": "700", "color": "#2d3436",
                               "marginTop": "24px", "marginBottom": "2px"}),
        html.P(subtitle, className="text-muted mb-2",
               style={"fontSize": "0.88rem"}) if subtitle else None,
    ])


def kpi_card(title, value, sub, border_color):
    return dbc.Card([dbc.CardBody([
        html.P(title, className="mb-1",
               style={"fontSize":"0.75rem","textTransform":"uppercase",
                      "letterSpacing":"0.05em","color":"#636e72","fontWeight":"600"}),
        html.H4(value, className="mb-1", style={"fontWeight":"700","color":"#2d3436"}),
        html.P(sub, className="mb-0",
               style={"fontSize":"0.78rem","color":"#636e72"}),
    ])], style={
        "borderTop": f"4px solid {border_color}",
        "borderRadius": "8px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
        "transition": "box-shadow 0.2s",
    })


def tooltip_label(label, tip, uid):
    return html.Div([
        dbc.Label([label,
            html.Span(" ⓘ", id=f"tt-{uid}",
                style={"cursor":"pointer","color":"#7f8c8d","fontSize":"0.85rem"})]),
        dbc.Tooltip(tip, target=f"tt-{uid}", placement="right"),
    ])


def inp_row(children):
    return dbc.Row(children, className="mb-2")


# Model information tooltips
MODEL_INFO = {
    "lr": {
        "name": "Logistic Regression",
        "how": (
            "Fits a linear equation to the log-odds of default. Each feature gets a coefficient "
            "that represents how much a one-unit change in that variable shifts the log-odds of defaulting. "
            "Uses gradient descent to find the best coefficients."
        ),
        "read": (
            "Output is a probability between 0 and 1. The ROC curve shows performance at every threshold. "
            "Coefficients are directly interpretable: positive = increases default risk, negative = decreases it. "
            "Requires feature scaling (StandardScaler) — without it, variables on different scales dominate."
        ),
        "pros": "Fully transparent, fast to train, coefficients are directly interpretable, industry standard in regulated lending.",
        "cons": "Assumes a linear relationship between features and log-odds — cannot capture non-linear patterns or interactions.",
    },
    "dt": {
        "name": "Decision Tree",
        "how": (
            "Learns a series of yes/no rules by recursively splitting the data. At each node it finds "
            "the feature and threshold that best separates defaults from non-defaults (measured by Gini impurity). "
            "The result is a tree of if/else rules."
        ),
        "read": (
            "Each path from root to leaf is a rule: e.g. 'int_rate > 15% AND fico_avg < 650 → Default'. "
            "max_depth controls complexity — shallow trees are interpretable but underfit, "
            "deep trees overfit. The gap between train AUC and test AUC reveals overfitting."
        ),
        "pros": "Fully transparent decision rules, no scaling needed, easy to explain to non-technical stakeholders.",
        "cons": "Single trees tend to overfit. A small change in training data can produce a very different tree (high variance).",
    },
    "rf": {
        "name": "Random Forest",
        "how": (
            "Trains 200 independent decision trees, each on a random sample of the data (bootstrap) "
            "and a random subset of features at each split (sqrt of total features). "
            "Final prediction is the average of all 200 trees' probabilities."
        ),
        "read": (
            "The averaging of many diverse trees reduces variance dramatically compared to a single tree. "
            "Feature importance measures how much each variable reduces Gini impurity across all trees. "
            "Does not require feature scaling. More robust to outliers than Logistic Regression."
        ),
        "pros": "Robust to overfitting, handles non-linear interactions, naturally robust to multicollinearity, no scaling needed.",
        "cons": "Black box — individual trees can be visualised but the ensemble cannot. Slower to train than a single tree.",
    },
    "xgb": {
        "name": "XGBoost",
        "how": (
            "Builds trees sequentially: each new tree focuses on correcting the errors of the previous ones "
            "(gradient boosting). Uses a learning rate of 0.05 with 300 trees — many small steps rather than few large ones. "
            "subsample=0.8 and colsample_bytree=0.8 add randomness to reduce overfitting."
        ),
        "read": (
            "Outputs a probability. Feature importance reflects how often each variable is used for splitting "
            "and how much it reduces the loss. SHAP values provide the most complete picture of what each feature contributes. "
            "scale_pos_weight corrects class imbalance by giving more weight to the minority class (defaults)."
        ),
        "pros": "Best predictive performance on structured/tabular data, handles missing values, built-in regularisation.",
        "cons": "Many hyperparameters to tune, sequential training is slower than Random Forest, less interpretable without SHAP.",
    },
}


def model_info_badge(model_key):
    """Small info icon with a tooltip explaining how the model works and how to read it."""
    m = MODEL_INFO[model_key]
    tip_content = html.Div([
        html.Strong(m["name"], style={"fontSize":"0.95rem"}),
        html.Hr(style={"margin":"6px 0"}),
        html.P([html.Strong("How it works: "), m["how"]], style={"fontSize":"0.82rem","marginBottom":"6px"}),
        html.P([html.Strong("How to read results: "), m["read"]], style={"fontSize":"0.82rem","marginBottom":"6px"}),
        html.P([html.Span("[+] ", style={"color":"#27ae60"}), html.Strong("Pros: "), m["pros"]], style={"fontSize":"0.82rem","marginBottom":"4px"}),
        html.P([html.Span("[-] ", style={"color":"#e67e22"}), html.Strong("Cons: "), m["cons"]], style={"fontSize":"0.82rem","marginBottom":"0"}),
    ], style={"maxWidth":"420px","padding":"4px"})
    uid = f"model-info-{model_key}"
    return html.Span([
        html.Span(" ℹ️", id=uid,
            style={"cursor":"pointer","fontSize":"1rem","verticalAlign":"middle"}),
        dbc.Tooltip(tip_content, target=uid, placement="right",
            style={"maxWidth":"440px"},
            trigger="hover"),
    ])

# ─────────────────────────────────────────────────────────────────
# 6. TAB 1 — EDA
# ─────────────────────────────────────────────────────────────────
DIST_OPTIONS = [
    {"label":"FICO Score","value":"fico_avg"},
    {"label":"Interest Rate (%)","value":"int_rate"},
    {"label":"DTI Ratio","value":"dti"},
    {"label":"Annual Income (USD)","value":"annual_inc"},
    {"label":"Revolving Utilization (%)","value":"revol_util"},
    {"label":"Loan Amount (USD)","value":"loan_amnt"},
]

overview_cards = dbc.Row([
    dbc.Col(kpi_card("Dataset","Lending Club 2007–2018",
        f"First {len(df):,} loans with known outcome","#3498db"), md=3),
    dbc.Col(kpi_card("Features used","28 variables",
        "31 selected · 3 derived · origination-only","#27ae60"), md=3),
    dbc.Col(kpi_card("Models trained","4 classifiers",
        "Logistic Reg · Decision Tree · Random Forest · XGBoost","#9b59b6"), md=3),
    dbc.Col(kpi_card("Key challenge","Class imbalance",
        f"~{1-AVG_DEFAULT:.0%} no default vs ~{AVG_DEFAULT:.0%} default","#e67e22"), md=3),
], className="mb-4")

tab_eda = dbc.Container([
    overview_cards,

    section_header("About this Dataset",
        "Before any analysis, it is important to understand what data we are working with and how it was prepared."),
    concept_box("Dataset: Lending Club 2007–2018",
        f"The original dataset contains approximately 2 million loans and 151 columns. "
        f"For this project we work with a cleaned sample of {len(df):,} loans. "
        f"All loans with status 'Current' were excluded because their final outcome is still unknown at the time of analysis — "
        "including them would introduce noise into the target variable. "
        "The remaining loans all have a definitive outcome: either Fully Paid (no default) or a terminal delinquency state (default). "
        "The data spans originations from 2007 to Q4 2018, covering multiple economic cycles including the 2008 financial crisis."),
    concept_box("Why only 31 of 151 columns?",
        "The original dataset contains many columns that only exist after the loan has been active for some time — "
        "for example: total_pymnt (total amount paid), recoveries (amount recovered after charge-off), "
        "last_pymnt_amnt (last payment received). "
        "Using these variables to predict default would be circular reasoning: "
        "you can only know total_pymnt after you already know whether the borrower defaulted. "
        "This is called data leakage. "
        "The 31 columns we selected are all known at the moment of loan origination — "
        "the only point in time when a real model would need to make a decision."),
    concept_box("What is data leakage?",
        "Data leakage occurs when the model is trained on information that would not be available at prediction time. "
        "A model with leakage will appear to perform extremely well in testing but will fail completely in production. "
        "For example: if we included 'recoveries' as a feature, the model would learn that recoveries > 0 means "
        "the loan defaulted — which is true, but only because recoveries are recorded after default happens. "
        "The model would be learning the outcome, not predicting it. "
        "Detecting and eliminating leakage is one of the most critical steps in any real-world ML project."),
    html.Hr(),

    section_header("Class Imbalance",
        "Understanding the balance between outcomes is the first step before any analysis."),
    concept_box("What is class imbalance?",
        "When one outcome is much more frequent than the other. Here ~80% of loans are repaid and ~20% default. "
        "A model that always predicts 'No Default' would get 80% accuracy but catch zero defaults. "
        "That is why we use AUC-ROC instead of accuracy as our evaluation metric."),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_status), md=7),
        dbc.Col(dcc.Graph(figure=fig_pie),    md=5),
    ], className="mb-1"),
    insight_box(f"The dataset has a {AVG_DEFAULT:.1%} default rate — 1 in every 5 loans was not fully repaid. "
                "This imbalance must be handled explicitly in every model we train."),
    html.Hr(),

    section_header("Default Rate by Categorical Variables",
        "If default rate differs significantly across categories, that variable has predictive power."),
    concept_box("How to read these charts",
        "Each bar shows the fraction of loans in that category that ended in default. "
        "The dashed line is the overall average. Bars above the line = above-average risk."),
    dbc.Row([
        dbc.Col([dcc.Graph(figure=fig_grade),
            insight_box("Grade is the strongest single predictor. "
                "Grade G has ~6× more defaults than Grade A.", C["red"])], md=6),
        dbc.Col([dcc.Graph(figure=fig_purpose),
            insight_box("small_business is the riskiest purpose. "
                "wedding and major_purchase have the lowest default rates.", C["blue"])], md=6),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col([dcc.Graph(figure=fig_home),
            insight_box("Hypothesis confirmed: OWN < MORTGAGE < RENT. "
                "Owning a home signals financial stability and accumulated assets.")], md=6),
        dbc.Col([dcc.Graph(figure=fig_verif),
            insight_box("Counterintuitive: 'Verified' borrowers default MORE. "
                "This is a confounder — Lending Club verifies the riskiest applicants selectively, "
                "not a random sample. The variable is useful but cannot be interpreted causally.",
                C["orange"])], md=6),
    ], className="mb-2"),
    html.Hr(),

    section_header("Distributions of Numerical Variables",
        "The more separated the two curves, the more predictive power the variable has."),
    concept_box("How to read KDE / histogram plots",
        "Green = fully repaid loans. Red = defaulted loans. "
        "When distributions overlap heavily, the model will struggle to use that variable. "
        "When they are clearly offset, the variable is a strong predictor."),
    dbc.Row([dbc.Col([
        html.Label("Select variable to explore:"),
        dcc.Dropdown(id="dist-variable-dropdown", options=DIST_OPTIONS,
            value="fico_avg", clearable=False, style={"width":"320px"}),
    ], md=4)], className="mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="dist-hist-graph"), md=6),
        dbc.Col(dcc.Graph(id="dist-box-graph"),  md=6),
    ], className="mb-1"),
    insight_box("FICO score, interest rate, and DTI show the clearest separation. "
                "Loan amount has heavy overlap — it alone cannot distinguish risky from safe loans."),
    html.Hr(),

    section_header("Correlations",
        "Which variables are most related to default? And which are correlated with each other?"),
    concept_box("What is multicollinearity?",
        "When two features are highly correlated (e.g., int_rate and grade_num at ~0.9), "
        "they carry nearly identical information. This can inflate coefficients in Logistic Regression. "
        "Tree-based models (Random Forest, XGBoost) are naturally robust to this."),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_heatmap),  md=7),
        dbc.Col(dcc.Graph(figure=fig_corr_bar), md=5),
    ], className="mb-1"),
    insight_box("Top predictors: int_rate (+), grade_num (+), fico_avg (−), revol_util (+), dti (+). "
                "int_rate and grade_num are ~0.9 correlated — multicollinearity to watch in Logistic Regression."),
    html.Hr(),

    # ── NEW: Time series ──────────────────────────────────────────
    section_header("Default Rate & Loan Volume over Time",
        "How has credit risk evolved across economic cycles? The 2008 financial crisis left a visible mark."),
    concept_box("Why does vintage matter?",
        "Loans originated in different years face different economic conditions. "
        "A model trained only on 2007-2009 loans (crisis years) would overestimate risk for 2014-2016 originations. "
        "Understanding vintage effects is critical for model stability over time."),
    dbc.Row([dbc.Col([
        html.Label("Year range:"),
        dcc.RangeSlider(
            id="ts-year-slider",
            min=2007, max=2018, step=1, value=[2007, 2018],
            marks={y: str(y) for y in range(2007, 2019)},
            tooltip={"placement":"bottom","always_visible":False}),
    ], md=8)], className="mb-2"),
    dbc.Alert([
        html.Strong("Column issue_d not found in lending_club_sample.csv.  "),
        "To enable this chart: add ",
        html.Code("'issue_d'"),
        " to the usecols list in your EDA notebook and run it again to regenerate the CSV. Then restart the dashboard.",
    ], id="ts-missing-alert", color="warning",
       style={"display":"none"}, className="mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="ts-volume-graph"),       md=6),
        dbc.Col(dcc.Graph(id="ts-default-rate-graph"), md=6),
    ], className="mb-1"),
    insight_box("Loans originated in 2007-2009 show dramatically higher default rates — the fingerprint of the financial crisis. "
                "Post-2013 vintages are cleaner, but be cautious: some 2017-2018 loans may not have reached their final "
                "status yet, potentially understating their true default rate.",
                "#d63031"),
    html.Hr(),

    # ── NEW: Geographic heatmap ───────────────────────────────────
    section_header("Geographic Distribution — Default Rate by State",
        "Does geography predict credit risk? Some states have structurally higher default rates due to "
        "local economic conditions, lending regulations, and borrower demographics."),
    concept_box("How to read this map",
        "Color intensity represents the default rate of loans originated by borrowers in each state. "
        "States with very few loans are filtered out by the minimum volume slider below — "
        "a state with only 5 loans could show 0% or 100% default rate purely by chance."),
    dbc.Row([dbc.Col([
        html.Label("Minimum loans per state:"),
        dcc.Slider(
            id="geo-min-loans-slider",
            min=50, max=2000, step=50, value=200,
            marks={50:"50",500:"500",1000:"1,000",2000:"2,000"},
            tooltip={"placement":"bottom","always_visible":True}),
    ], md=7)], className="mb-2"),
    dcc.Graph(id="geo-map-graph"),
    insight_box("States in the South and Midwest tend to show above-average default rates. "
                "California, New York, and Texas dominate loan volume but show near-average risk — "
                "their large sample sizes make them the most statistically reliable data points."),
    html.Hr(),

    # ── NEW: Sub-grade heatmap ────────────────────────────────────
    section_header("Sub-Grade Risk Heatmap (A1 to G5)",
        "Lending Club assigns 35 sub-grades, not just 7 grades. Each sub-grade corresponds to a specific "
        "interest rate band. This heatmap shows how default risk increases within each grade."),
    concept_box("Why sub-grade matters for modeling",
        "Using sub_grade instead of grade gives the model 5x more granularity on the risk dimension. "
        "The difference between B1 and B5 can be 5-8 percentage points in default rate. "
        "If your dataset includes sub_grade, it is almost always a stronger predictor than grade alone."),
    dcc.Graph(id="subgrade-heatmap-graph"),
    insight_box("The heatmap should show a clean gradient from top-left (A1, lowest risk) to bottom-right (G5, highest risk). "
                "Any cell that breaks this monotonic pattern is worth investigating — "
                "it may indicate thin data or a structural anomaly in Lending Club's grading model."),
    html.Hr(),

    # ── NEW: Loan amount by purpose ───────────────────────────────
    section_header("Loan Amount by Purpose",
        "Do borrowers request different amounts for different purposes? And does loan size interact with default risk?"),
    concept_box("How to read this chart",
        "Each box shows the distribution of loan amounts for that purpose. "
        "The color of each box represents the default rate for that purpose — "
        "so you can simultaneously see the typical loan size and the risk level."),
    dbc.Row([dbc.Col([
        html.Label("Loan amount range ($):"),
        dcc.RangeSlider(
            id="purpose-amount-slider",
            min=500, max=40000, step=500, value=[500, 40000],
            marks={500:"$500",10000:"$10k",20000:"$20k",35000:"$35k",40000:"$40k"},
            tooltip={"placement":"bottom","always_visible":False}),
    ], md=9)], className="mb-2"),
    dcc.Graph(id="purpose-amount-graph"),
    insight_box("small_business loans are both larger than average and riskier. "
                "debt_consolidation is the most common purpose and shows near-average risk — "
                "it dominates the dataset and 'anchors' the baseline default rate."),

], fluid=True)

# ─────────────────────────────────────────────────────────────────
# 7. TAB 2 — MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────
tab_model = dbc.Container([

    section_header("Model Comparison",
        "Four models with increasing complexity to understand the trade-off between power and interpretability."),
    html.Div([
        html.P("Metrics computed on a stratified hold-out test set. "
               "Bold green = best value in column. XGBoost uses the optimal F1 threshold; all others use 0.5.",
               style={"fontSize":"0.83rem","color":"#636e72","marginBottom":"8px"}),
        summary_table,
    ], className="mb-3"),
    concept_box("AUC-ROC and Gini — why these metrics?",
        "Accuracy is misleading with imbalanced data: a model that always predicts 'No Default' gets 80% accuracy "
        "but catches zero defaults. AUC-ROC measures how well the model ranks borrowers by risk at every threshold. "
        "AUC = 0.5 means random; AUC = 1.0 means perfect separation. "
        "Gini = 2×AUC − 1 is the same metric rescaled to [0,1] and is the standard in retail banking credit scoring."),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Strong("Models: "),
                model_info_badge("lr"), html.Span(" Logistic Regression  ", style={"fontSize":"0.9rem"}),
                model_info_badge("dt"), html.Span(" Decision Tree  ",        style={"fontSize":"0.9rem"}),
                model_info_badge("rf"), html.Span(" Random Forest  ",        style={"fontSize":"0.9rem"}),
                model_info_badge("xgb"), html.Span(" XGBoost",               style={"fontSize":"0.9rem"}),
            ], className="mb-2"),
        ]),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_summary), md=6),
        dbc.Col(dcc.Graph(figure=fig_roc),     md=6),
    ], className="mb-1"),
    insight_box("Ranking: Decision Tree < Logistic Regression < Random Forest < XGBoost. "
                "A well-engineered linear model already beats a single tree. "
                "The additional gain from ensembles comes from capturing non-linear interactions."),
    html.Hr(),

    section_header("Precision vs Recall — The Business Trade-off",
        "The Precision-Recall curve focuses on the positive class (defaults) and is more informative than ROC with imbalanced data."),
    concept_box("Precision vs Recall",
        "Precision: of all loans flagged as risky, what fraction actually defaults? "
        "Recall: of all loans that default, what fraction does the model catch? "
        "These goals are always in tension — improving one hurts the other. "
        "The right balance is a business decision: how costly is a missed default vs a rejected good borrower?"),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_pr), md=6),
        dbc.Col(dcc.Graph(figure=fig_f1), md=6),
    ], className="mb-1"),
    insight_box(f"The optimal threshold for XGBoost (maximizing F1) is {OPTIMAL_THRESHOLD:.2f}, not 0.5. "
                "With imbalanced data, 0.5 is almost never optimal — "
                "defaults cluster in the 0.2–0.4 probability range and get missed at threshold 0.5."),
    html.Hr(),

    section_header("Confusion Matrix — Threshold Sensitivity (XGBoost)",
        "Move the slider to see how the decision threshold affects false positives vs false negatives."),
    concept_box("What is the decision threshold?",
        "The model outputs a probability between 0 and 1. The threshold is the cutoff: above it we predict Default. "
        "Lower threshold → catches more defaults (higher recall) but flags more good borrowers (lower precision). "
        "Higher threshold → opposite. The correct threshold depends on the bank's cost structure."),
    dbc.Row([dbc.Col([
        html.Label("Decision threshold:"),
        dcc.Slider(id="threshold-slider", min=0.1, max=0.9, step=0.05, value=OPTIMAL_THRESHOLD,
            marks={round(v,2):str(round(v,2)) for v in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]},
            tooltip={"placement":"bottom","always_visible":True}),
    ], md=8)], className="mb-2 mt-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="cm-graph"), md=8),
        dbc.Col(html.Div(id="cm-metrics"), md=4),
    ], className="mb-1"),
    insight_box("False Negatives (missed defaults) = financial loss for the bank. "
                "False Positives (good loans rejected) = lost revenue. "
                "Lowering the threshold reduces FN but increases FP. Neither is free."),
    html.Hr(),

    section_header("Feature Importance",
        "Which variables does the model rely on most?"),
    dbc.Row([dbc.Col([
        html.Div([
            dbc.RadioItems(id="fi-model-radio",
                options=[
                    {"label": html.Span(["Random Forest ", model_info_badge("rf")]), "value":"rf"},
                    {"label": html.Span(["XGBoost ",       model_info_badge("xgb")]), "value":"xgb"},
                ],
                value="xgb", inline=True, className="mb-2"),
        ]),
        dcc.Graph(id="fi-graph"),
    ])], className="mb-1"),
    insight_box("Both models agree: int_rate, grade_num, fico_avg, dti are the top features. "
                "Consensus between two independent algorithms is strong evidence of genuine predictive signal."),
    html.Hr(),

    section_header("SHAP Values — Model Interpretability",
        "Feature importance tells us WHICH variables matter. SHAP tells us HOW they matter and explains each individual prediction."),
    concept_box("What are SHAP values?",
        "SHAP (SHapley Additive exPlanations) comes from game theory. It computes the contribution of each feature "
        "to each individual prediction — how much each variable pushed the score above or below the baseline. "
        "In the beeswarm: each dot is one loan, x-axis = impact, color = feature value (red=high, blue=low). "
        "In waterfall plots: red bars pushed toward default, blue bars pushed away. "
        "This individual-level transparency is required by financial regulators (EU AI Act, ECOA): "
        "applicants denied credit have a legal right to a meaningful explanation, and SHAP makes that possible."),
    dbc.Row([dbc.Col([
        dbc.Button("Load SHAP Plots", id="shap-load-btn", color="primary",
                   outline=True, className="mb-3"),
        html.Small(" (click to generate — takes ~10 seconds)",
                   className="text-muted"),
    ])], className="mb-2"),
    dbc.Row([
        dbc.Col(html.Img(id="shap-beeswarm-img", style={"width":"100%"}), md=6),
        dbc.Col(html.Img(id="shap-bar-img",      style={"width":"100%"}), md=6),
    ], className="mb-2"),
    insight_box("High int_rate (red, right) increases default risk. High fico_avg (red, left) decreases it. "
                "All directions match economic intuition — a model with contradictory SHAP directions "
                "would be a red flag for data leakage.", C["blue"]),
    dbc.Row([
        dbc.Col([html.H6("Case A — High-Risk Borrower", className="text-center text-danger"),
                 html.Img(id="shap-waterfall-high-img", style={"width":"100%"})], md=6),
        dbc.Col([html.H6("Case B — Low-Risk Borrower", className="text-center text-success"),
                 html.Img(id="shap-waterfall-low-img",  style={"width":"100%"})], md=6),
    ], className="mb-2"),
    insight_box("Case A: no single factor disqualifies — it is the combination of high int_rate + "
                "high grade_num + low fico_avg that creates the high-risk profile. "
                "Case B: a high FICO score, low rate, and manageable DTI all push well below the baseline."),

    section_header("Risk Decile Analysis",
        "A well-calibrated model should rank borrowers correctly: riskiest decile = highest actual default rate."),
    concept_box("What is a decile analysis?",
        "Borrowers are sorted by predicted probability and split into 10 equal groups. "
        "Decile 1 = lowest predicted risk, Decile 10 = highest. "
        "A good model produces a monotonically increasing actual default rate from left to right. "
        "This is the standard evaluation method in retail banking (also called a 'lift table')."),
    dcc.Graph(figure=fig_decile),
    insight_box("Default rate increases consistently from Decile 1 to Decile 10. "
                "A bank could directly operationalise this: auto-approve Deciles 1–4, "
                "manual review for 5–7, auto-decline Deciles 8–10."),
], fluid=True)

# ─────────────────────────────────────────────────────────────────
# 8. TAB 3 — SCORE CALCULATOR
# ─────────────────────────────────────────────────────────────────
PURPOSES  = sorted(df["purpose"].dropna().unique().tolist())
HOME_OPTS = ["OWN","MORTGAGE","RENT"]

TIPS = {
    "loan_amnt":   "Total amount of money requested by the borrower.",
    "term":        "Repayment period. 60-month loans tend to have higher default rates than 36-month.",
    "grade":       "Risk grade assigned by Lending Club (A = safest, G = riskiest). Derived from credit history, DTI, etc.",
    "purpose":     "Stated reason for the loan. small_business tends to be the riskiest purpose.",
    "int_rate":    "Annual interest rate. Lending Club sets this based on the grade — higher rate = riskier borrower.",
    "annual_inc":  "Self-reported annual income (or combined for joint applications).",
    "home":        "Housing situation. Homeowners (OWN) default less than renters (RENT).",
    "dti":         "Debt-to-Income ratio: total monthly debt ÷ monthly income × 100. Higher = more financial stress.",
    "fico":        "FICO credit score at origination (300–850). Strong negative predictor: higher FICO = lower default risk.",
    "revol_util":  "Revolving utilization: % of available revolving credit currently used. High utilization signals financial stress.",
    "emp_length":  "Years at current employer. Proxy for income stability.",
    "installment": "Monthly payment amount. Used to compute installment-to-income ratio.",
    "open_acc":    "Number of currently open credit accounts.",
    "pub_rec":     "Derogatory public records (bankruptcies, tax liens, judgements). Even 1 is a negative signal.",
    "revol_bal":   "Total current revolving balance across all credit accounts.",
    "delinq_2yrs": "Times 30+ days past due in the last 2 years. Any value > 0 is a risk flag.",
    "inq_6mths":   "Hard credit inquiries in the last 6 months. Many inquiries signal the borrower is seeking a lot of credit.",
    "mort_acc":    "Number of mortgage accounts. Can signal financial sophistication or over-leverage.",
    "bc_util":     "Bankcard utilization rate. Similar to revol_util but specific to bank-issued credit cards.",
    "threshold":   f"Probability cutoff above which the loan is classified as Default. "
                   f"The statistically optimal value (max F1) is {OPTIMAL_THRESHOLD:.2f}. "
                   "Default of 0.5 often misses many defaults with imbalanced data.",
}

tab_calc = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H5("Borrower & Loan Details", className="mb-3"),
            concept_box("How this works",
                "Fill in the details below and click 'Calculate Risk'. "
                "The XGBoost model trained on 133,000+ Lending Club loans will output the predicted default probability. "
                "Hover over ⓘ next to each label for an explanation. "
                "Missing inputs are filled with training set medians."),
            dbc.Button("Calculate Risk", id="calc-button", color="primary",
                       size="lg", className="w-100 mb-3"),

            inp_row([
                dbc.Col([tooltip_label("Loan Amount ($)", TIPS["loan_amnt"], "loan-amnt"),
                    dbc.Input(id="inp-loan-amnt", type="number", value=10000, min=500, max=40000, step=500)], md=6),
                dbc.Col([tooltip_label("Term", TIPS["term"], "term"),
                    dcc.Dropdown(id="inp-term",
                        options=[{"label":"36 months","value":36},{"label":"60 months","value":60}],
                        value=36, clearable=False)], md=6),
            ]),
            inp_row([
                dbc.Col([tooltip_label("Grade", TIPS["grade"], "grade"),
                    dcc.Dropdown(id="inp-grade",
                        options=[{"label":g,"value":g} for g in ["A","B","C","D","E","F","G"]],
                        value="C", clearable=False)], md=6),
                dbc.Col([tooltip_label("Purpose", TIPS["purpose"], "purpose"),
                    dcc.Dropdown(id="inp-purpose",
                        options=[{"label":p,"value":p} for p in PURPOSES],
                        value="debt_consolidation", clearable=False)], md=6),
            ]),
            inp_row([dbc.Col([
                tooltip_label("Interest Rate (%)", TIPS["int_rate"], "int-rate"),
                dcc.Slider(id="inp-int-rate", min=5, max=30, step=0.5, value=12,
                    marks={5:"5%",15:"15%",30:"30%"},
                    tooltip={"placement":"bottom","always_visible":True}),
            ])]),
            inp_row([
                dbc.Col([tooltip_label("Annual Income ($)", TIPS["annual_inc"], "annual-inc"),
                    dbc.Input(id="inp-annual-inc", type="number", value=60000, min=10000, max=500000, step=1000)], md=6),
                dbc.Col([tooltip_label("Home Ownership", TIPS["home"], "home"),
                    dcc.Dropdown(id="inp-home",
                        options=[{"label":h,"value":h} for h in HOME_OPTS],
                        value="MORTGAGE", clearable=False)], md=6),
            ]),
            inp_row([dbc.Col([
                tooltip_label("DTI Ratio", TIPS["dti"], "dti"),
                dcc.Slider(id="inp-dti", min=0, max=50, step=0.5, value=15,
                    marks={0:"0",25:"25",50:"50"},
                    tooltip={"placement":"bottom","always_visible":True}),
            ])]),
            inp_row([dbc.Col([
                tooltip_label("FICO Score", TIPS["fico"], "fico"),
                dcc.Slider(id="inp-fico", min=580, max=850, step=5, value=700,
                    marks={580:"580",700:"700",850:"850"},
                    tooltip={"placement":"bottom","always_visible":True}),
            ])]),
            inp_row([dbc.Col([
                tooltip_label("Revolving Utilization (%)", TIPS["revol_util"], "revol-util"),
                dcc.Slider(id="inp-revol-util", min=0, max=100, step=1, value=40,
                    marks={0:"0%",50:"50%",100:"100%"},
                    tooltip={"placement":"bottom","always_visible":True}),
            ])]),
            inp_row([
                dbc.Col([tooltip_label("Employment Length (yrs)", TIPS["emp_length"], "emp-length"),
                    dcc.Dropdown(id="inp-emp-length",
                        options=[{"label":f"{i} yr{'s' if i!=1 else ''}","value":i} for i in range(11)],
                        value=5, clearable=False)], md=6),
                dbc.Col([tooltip_label("Installment ($/month)", TIPS["installment"], "installment"),
                    dbc.Input(id="inp-installment", type="number", value=300, min=10, max=3000, step=10)], md=6),
            ]),
            inp_row([
                dbc.Col([tooltip_label("Open Accounts", TIPS["open_acc"], "open-acc"),
                    dbc.Input(id="inp-open-acc", type="number", value=10, min=0, max=60, step=1)], md=6),
                dbc.Col([tooltip_label("Public Records", TIPS["pub_rec"], "pub-rec"),
                    dbc.Input(id="inp-pub-rec", type="number", value=0, min=0, max=10, step=1)], md=6),
            ]),
            inp_row([
                dbc.Col([tooltip_label("Revolving Balance ($)", TIPS["revol_bal"], "revol-bal"),
                    dbc.Input(id="inp-revol-bal", type="number", value=15000, min=0, max=500000, step=1000)], md=6),
                dbc.Col([tooltip_label("Delinquencies (2yr)", TIPS["delinq_2yrs"], "delinq"),
                    dbc.Input(id="inp-delinq", type="number", value=0, min=0, max=20, step=1)], md=6),
            ]),
            inp_row([
                dbc.Col([tooltip_label("Inquiries (6 months)", TIPS["inq_6mths"], "inq"),
                    dbc.Input(id="inp-inq", type="number", value=0, min=0, max=10, step=1)], md=6),
                dbc.Col([tooltip_label("Mortgage Accounts", TIPS["mort_acc"], "mort-acc"),
                    dbc.Input(id="inp-mort-acc", type="number", value=1, min=0, max=20, step=1)], md=6),
            ]),
            inp_row([dbc.Col([
                tooltip_label("Bankcard Utilization (%)", TIPS["bc_util"], "bc-util"),
                dcc.Slider(id="inp-bc-util", min=0, max=100, step=1, value=45,
                    marks={0:"0%",50:"50%",100:"100%"},
                    tooltip={"placement":"bottom","always_visible":True}),
            ])]),
            html.Hr(),
            inp_row([dbc.Col([
                tooltip_label(f"Decision Threshold (optimal = {OPTIMAL_THRESHOLD:.2f})",
                              TIPS["threshold"], "threshold"),
                dcc.Slider(id="inp-threshold", min=0.1, max=0.9, step=0.05, value=OPTIMAL_THRESHOLD,
                    marks={0.1:"0.1",0.3:"0.3",0.5:"0.5",0.7:"0.7",0.9:"0.9"},
                    tooltip={"placement":"bottom","always_visible":True}),
            ])]),
        ], md=5, style={"background":"#f8f9fa","padding":"20px","borderRadius":"8px"}),

        dbc.Col([
            html.Div(id="calc-output",
                children=dbc.Alert("Fill in the form and click 'Calculate Risk' to get the prediction.",
                                   color="secondary")),
            html.Hr(),
            dcc.Graph(id="calc-dist-graph"),
            html.Hr(),
            html.H6("What drove this prediction? (SHAP waterfall)", className="mt-2"),
            html.Small("Red bars pushed toward default. Blue bars pushed away from default.",
                       className="text-muted"),
            html.Br(),
            html.Img(id="calc-shap-img", style={"width":"100%"}),
        ], md=7),
    ], className="mt-3"),
], fluid=True)

# ─────────────────────────────────────────────────────────────────
# 9. MAIN APP
# ─────────────────────────────────────────────────────────────────
app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True)
app.title = "Lending Club — Credit Risk"
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background-color: #f0f3f7 !important; font-family: 'Inter', 'Segoe UI', sans-serif; }
            .nav-tabs .nav-link { font-weight: 600; font-size: 0.92rem; color: #636e72; letter-spacing: 0.02em; }
            .nav-tabs .nav-link.active { color: #0984e3 !important; border-bottom: 3px solid #0984e3; background: white; }
            .card { border: none !important; border-radius: 8px !important; }
            .dash-graph { border-radius: 8px; overflow: hidden; }
            hr { border-color: #dfe6e9; margin: 24px 0; }
            .dashboard-header { background: linear-gradient(135deg, #2d3436 0%, #0984e3 100%);
                color: white; padding: 20px 28px; border-radius: 10px; margin-bottom: 20px; }
            .dashboard-header h2 { color: white !important; font-weight: 700; margin-bottom: 4px; }
            .dashboard-header small { color: rgba(255,255,255,0.75) !important; }
            .footer-bar { background: #2d3436; color: rgba(255,255,255,0.65);
                font-size: 0.8rem; padding: 14px 24px; border-radius: 8px;
                margin-top: 32px; margin-bottom: 16px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = dbc.Container([
    html.Div([
        html.H2("Lending Club — Credit Risk & Loan Default Prediction"),
        html.Small(
            "Data Business in Python · Final Project · "
            "Logistic Regression · Decision Tree · Random Forest · XGBoost · SHAP"
        ),
    ], className="dashboard-header"),
    dbc.Tabs([
        dbc.Tab(tab_eda,   label="EDA",             tab_id="tab-eda"),
        dbc.Tab(tab_model, label="Model Evaluation", tab_id="tab-model"),
        dbc.Tab(tab_calc,  label="Score Calculator", tab_id="tab-calc"),
    ], id="tabs", active_tab="tab-eda"),
    html.Div([
        html.Span("Data source: "),
        html.A("Lending Club Loan Data 2007–2018 (Kaggle)",
               href="https://www.kaggle.com/datasets/wordsforthewise/lending-club",
               style={"color":"rgba(255,255,255,0.75)","textDecoration":"underline"}),
        html.Span("  ·  Loans excluded: 'Current' status (outcome unknown)  ·  "
                  "Features: origination-only (no post-origination leakage)  ·  "
                  "Models evaluated on a stratified hold-out test set  ·  "
                  "Academic project — not financial advice"),
    ], className="footer-bar"),
], fluid=True)

# ─────────────────────────────────────────────────────────────────
# 10. CALLBACKS
# ─────────────────────────────────────────────────────────────────
# ── Callback: Time series ──────────────────────────────────────────
HAS_ISSUE_D = "issue_d" in df.columns

@app.callback(
    Output("ts-volume-graph",       "figure"),
    Output("ts-default-rate-graph", "figure"),
    Output("ts-missing-alert",      "style"),
    Input("ts-year-slider",         "value"),
)
def update_timeseries(year_range):
    _hidden  = {"display":"none"}
    _visible = {"display":"block"}

    def _empty_fig(msg):
        f = go.Figure()
        f.update_layout(
            height=300, plot_bgcolor=C["bg"],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(
                text=msg, xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=13, color=C["grey"]))])
        return f

    if not HAS_ISSUE_D:
        msg = "Add issue_d to your EDA notebook usecols and regenerate lending_club_sample.csv"
        return _empty_fig(msg), _empty_fig(msg), _visible

    ts = df.copy()
    # Parse flexibly — handles "Jan-2015", "2015-01-01", "01/2015", etc.
    ts["issue_d_parsed"] = pd.to_datetime(ts["issue_d"], errors="coerce")
    ts = ts.dropna(subset=["issue_d_parsed"])
    ts["year"]    = ts["issue_d_parsed"].dt.year
    ts["quarter"] = ts["issue_d_parsed"].dt.quarter
    ts["period"]  = ts["year"].astype(str) + "-Q" + ts["quarter"].astype(str)
    ts = ts[(ts["year"] >= year_range[0]) & (ts["year"] <= year_range[1])]

    if ts.empty:
        msg = "No data found for the selected year range."
        return _empty_fig(msg), _empty_fig(msg), _hidden

    agg = ts.groupby("period").agg(
        count=("default","count"),
        default_rate=("default","mean")).reset_index()
    agg = agg.sort_values("period")

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=agg["period"], y=agg["count"],
        marker_color=C["blue"], opacity=0.8, name="Loan volume"))
    fig_vol.update_layout(
        title="Loan Volume per Quarter",
        xaxis_title="", yaxis_title="Number of loans",
        plot_bgcolor=C["bg"], xaxis_tickangle=-45, height=340)

    fig_dr = go.Figure()
    fig_dr.add_trace(go.Scatter(
        x=agg["period"], y=agg["default_rate"],
        mode="lines+markers",
        line=dict(color=C["red"], width=2),
        marker=dict(size=5),
        name="Default rate"))
    avg_dr = ts["default"].mean()
    fig_dr.add_hline(y=avg_dr, line_dash="dash", line_color=C["grey"],
        annotation_text=f"Period avg: {avg_dr:.1%}")
    fig_dr.update_layout(
        title="Default Rate per Quarter",
        xaxis_title="", yaxis_title="Default rate",
        yaxis_tickformat=".0%",
        plot_bgcolor=C["bg"], xaxis_tickangle=-45, height=340)

    return fig_vol, fig_dr, _hidden


# ── Callback: Geographic heatmap ───────────────────────────────────
@app.callback(
    Output("geo-map-graph", "figure"),
    Input("geo-min-loans-slider", "value"),
)
def update_geo(min_loans):
    if "addr_state" not in df.columns:
        empty = go.Figure()
        empty.update_layout(title="addr_state column not available in dataset")
        return empty

    geo = df.groupby("addr_state").agg(
        count=("default","count"),
        default_rate=("default","mean")).reset_index()
    geo = geo[geo["count"] >= min_loans]

    fig = px.choropleth(
        geo,
        locations="addr_state",
        locationmode="USA-states",
        color="default_rate",
        scope="usa",
        color_continuous_scale="RdYlGn_r",
        range_color=[geo["default_rate"].min(), geo["default_rate"].max()],
        hover_name="addr_state",
        hover_data={"count":True,"default_rate":":.1%"},
        labels={"default_rate":"Default Rate","count":"Loan count"},
        title=f"Default Rate by State (states with >= {min_loans} loans)")
    fig.update_layout(
        height=480,
        coloraxis_colorbar=dict(tickformat=".0%"),
        geo=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


# ── Callback: Sub-grade heatmap ────────────────────────────────────
@app.callback(
    Output("subgrade-heatmap-graph", "figure"),
    Input("tabs", "active_tab"),
)
def update_subgrade(_):
    if "sub_grade" not in df.columns:
        empty = go.Figure()
        empty.update_layout(title="sub_grade column not available in dataset")
        return empty

    sg = df.groupby("sub_grade").agg(
        count=("default","count"),
        default_rate=("default","mean")).reset_index()
    sg["grade"]  = sg["sub_grade"].str[0]
    sg["level"]  = sg["sub_grade"].str[1].astype(int)
    pivot_dr     = sg.pivot(index="grade",  columns="level", values="default_rate")
    pivot_count  = sg.pivot(index="grade",  columns="level", values="count")
    grade_order  = ["A","B","C","D","E","F","G"]
    pivot_dr     = pivot_dr.reindex(grade_order)
    pivot_count  = pivot_count.reindex(grade_order)

    text_matrix = []
    for grade in grade_order:
        row_text = []
        for lvl in [1,2,3,4,5]:
            try:
                dr  = pivot_dr.loc[grade, lvl]
                cnt = pivot_count.loc[grade, lvl]
                row_text.append(f"{dr:.1%}<br>n={int(cnt):,}")
            except Exception:
                row_text.append("")
        text_matrix.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=pivot_dr.values,
        x=[f"Level {i}" for i in [1,2,3,4,5]],
        y=grade_order,
        text=text_matrix,
        texttemplate="%{text}",
        colorscale="RdYlGn_r",
        colorbar=dict(tickformat=".0%", title="Default Rate"),
        hoverongaps=False))
    fig.update_layout(
        title="Default Rate by Sub-Grade (A1 to G5)",
        xaxis_title="Sub-grade level (1 = best, 5 = worst within grade)",
        yaxis_title="Grade",
        height=420,
        plot_bgcolor=C["bg"],
        yaxis=dict(autorange="reversed"))
    return fig


# ── Callback: Loan amount by purpose ──────────────────────────────
@app.callback(
    Output("purpose-amount-graph", "figure"),
    Input("purpose-amount-slider", "value"),
)
def update_purpose_amount(amount_range):
    filtered = df[
        (df["loan_amnt"] >= amount_range[0]) &
        (df["loan_amnt"] <= amount_range[1])
    ]
    purpose_dr = filtered.groupby("purpose")["default"].mean().reset_index()
    purpose_dr.columns = ["purpose","default_rate"]

    fig = px.box(
        filtered,
        x="purpose", y="loan_amnt",
        color="purpose",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={"loan_amnt":"Loan Amount ($)","purpose":""},
        title=f"Loan Amount Distribution by Purpose  (${amount_range[0]:,} – ${amount_range[1]:,})")
    fig.update_traces(showlegend=False)
    # Overlay default rate as scatter on secondary axis
    fig.update_layout(
        plot_bgcolor=C["bg"],
        xaxis_tickangle=-35,
        height=480,
        yaxis_title="Loan Amount ($)",
        yaxis=dict(tickformat="$,.0f"),
    )
    # Sort x by default rate
    dr_order = purpose_dr.sort_values("default_rate")["purpose"].tolist()
    fig.update_layout(xaxis=dict(categoryorder="array", categoryarray=dr_order))
    return fig


# ── Callback: SHAP lazy load ───────────────────────────────────────
@app.callback(
    Output("shap-beeswarm-img",      "src"),
    Output("shap-bar-img",           "src"),
    Output("shap-waterfall-high-img","src"),
    Output("shap-waterfall-low-img", "src"),
    Input("shap-load-btn",           "n_clicks"),
    prevent_initial_call=True,
)
def load_shap_plots(_):
    return SHAP_BEESWARM, SHAP_BAR, SHAP_WATERFALL_HIGH, SHAP_WATERFALL_LOW


# ── Callback: Distribution histograms ─────────────────────────────
@app.callback(
    Output("dist-hist-graph","figure"),
    Output("dist-box-graph", "figure"),
    Input("dist-variable-dropdown","value"),
)
def update_dist(col):
    try:
        label = next(o["label"] for o in DIST_OPTIONS if o["value"]==col)
        cap   = df[col].quantile(0.99)
        data  = df[df[col] <= cap].copy()
        # Sample to reduce memory pressure on Render Free
        if len(data) > 20_000:
            data = data.sample(20_000, random_state=42)
        fig_h = px.histogram(data, x=col, color="default_label", barmode="overlay",
            color_discrete_map={"No Default":C["green"],"Default":C["red"]},
            opacity=0.6, nbins=60, labels={col:label,"default_label":""},
            title=f"Distribution: {label}")
        fig_h.update_layout(plot_bgcolor=C["bg"])
        fig_b = px.box(data, x="default_label", y=col, color="default_label",
            color_discrete_map={"No Default":C["green"],"Default":C["red"]},
            labels={col:label,"default_label":""}, title=f"Boxplot: {label}")
        fig_b.update_layout(showlegend=False, plot_bgcolor=C["bg"])
        return fig_h, fig_b
    except Exception as e:
        empty = go.Figure()
        empty.update_layout(title=f"Error loading {col}: {str(e)}", plot_bgcolor=C["bg"])
        return empty, empty


@app.callback(
    Output("cm-graph","figure"),
    Output("cm-metrics","children"),
    Input("threshold-slider","value"),
)
def update_cm(threshold):
    y_pred = (proba_xgb >= threshold).astype(int)
    cm     = confusion_matrix(y_test, y_pred)
    rep    = classification_report(y_test, y_pred, output_dict=True)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
        x=["Predicted: No Default","Predicted: Default"],
        y=["Actual: No Default","Actual: Default"],
        title=f"Confusion Matrix — XGBoost (threshold = {threshold})", aspect="auto")
    fig.update_layout(height=380, coloraxis_showscale=False)
    tn, fp, fn, tp = cm.ravel()
    metrics = dbc.Table([html.Tbody([
        html.Tr([html.Td("Threshold"),                   html.Td(f"{threshold}")]),
        html.Tr([html.Td("Precision"),                   html.Td(f"{rep['1']['precision']:.3f}")]),
        html.Tr([html.Td("Recall"),                      html.Td(f"{rep['1']['recall']:.3f}")]),
        html.Tr([html.Td("F1-Score"),                    html.Td(f"{rep['1']['f1-score']:.3f}")]),
        html.Tr([html.Td("TP — caught defaults"),        html.Td(f"{tp:,}")]),
        html.Tr([html.Td("FP — good loans rejected"),    html.Td(f"{fp:,}")]),
        html.Tr([html.Td("TN — correct approvals"),      html.Td(f"{tn:,}")]),
        html.Tr([html.Td("FN — missed defaults"),        html.Td(f"{fn:,}")]),
    ])], bordered=True, size="sm", className="mt-3")
    return fig, metrics


@app.callback(
    Output("fi-graph","figure"),
    Input("fi-model-radio","value"),
)
def update_fi(model_choice):
    return fig_fi_xgb if model_choice=="xgb" else fig_fi_rf


@app.callback(
    Output("calc-output",    "children"),
    Output("calc-dist-graph","figure"),
    Output("calc-shap-img",  "src"),
    Input("calc-button","n_clicks"),
    State("inp-loan-amnt",  "value"),
    State("inp-term",       "value"),
    State("inp-grade",      "value"),
    State("inp-purpose",    "value"),
    State("inp-int-rate",   "value"),
    State("inp-annual-inc", "value"),
    State("inp-home",       "value"),
    State("inp-dti",        "value"),
    State("inp-fico",       "value"),
    State("inp-revol-util", "value"),
    State("inp-emp-length", "value"),
    State("inp-installment","value"),
    State("inp-open-acc",   "value"),
    State("inp-pub-rec",    "value"),
    State("inp-revol-bal",  "value"),
    State("inp-delinq",     "value"),
    State("inp-inq",        "value"),
    State("inp-mort-acc",   "value"),
    State("inp-bc-util",    "value"),
    State("inp-threshold",  "value"),
    prevent_initial_call=True,
)
def update_calculator(
    n_clicks,
    loan_amnt, term, grade, purpose, int_rate,
    annual_inc, home, dti, fico, revol_util,
    emp_length, installment, open_acc, pub_rec,
    revol_bal, delinq, inq, mort_acc, bc_util, threshold
):
    grade_map = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
    row = {col: TRAIN_MEDIANS.get(col, 0) for col in FEATURE_COLS}

    row["loan_amnt"]             = loan_amnt   or 10000
    row["term_months"]           = term        or 36
    row["int_rate"]              = int_rate    or 12
    row["installment"]           = installment or 300
    row["grade_num"]             = grade_map.get(grade, 3)
    row["annual_inc"]            = annual_inc  or 60000
    row["dti"]                   = dti         or 15
    row["fico_avg"]              = fico        or 700
    row["revol_util"]            = revol_util  or 40
    row["emp_length_yrs"]        = emp_length  or 5
    row["open_acc"]              = open_acc    or 10
    row["pub_rec"]               = pub_rec     or 0
    row["revol_bal"]             = revol_bal   or 15000
    row["delinq_2yrs"]           = delinq      or 0
    row["inq_last_6mths"]        = inq         or 0
    row["mort_acc"]              = mort_acc    or 1
    row["bc_util"]               = bc_util     or 45
    row["installment_to_income"] = (installment or 300) / ((annual_inc or 60000) / 12)
    row["credit_history_years"]  = TRAIN_MEDIANS.get("credit_history_years", 10)
    row["delinquency_flag"]      = 1 if (delinq or 0) > 0 else 0

    for opt in HOME_OPTS[1:]:
        key = f"home_ownership_{opt}"
        if key in row:
            row[key] = 1 if home==opt else 0
    for p in PURPOSES[1:]:
        key = f"purpose_{p}"
        if key in row:
            row[key] = 1 if purpose==p else 0

    X_input = pd.DataFrame([row])[FEATURE_COLS]
    X_input = X_input.fillna(0).replace([np.inf,-np.inf], 0)

    prob      = float(xgb.predict_proba(X_input)[0,1])
    threshold = threshold or OPTIMAL_THRESHOLD
    decision  = "DEFAULT" if prob >= threshold else "APPROVED"

    if prob < 0.15:
        risk_label, risk_color, risk_emoji = "LOW RISK",    C["green"],  "✅"
    elif prob < 0.35:
        risk_label, risk_color, risk_emoji = "MEDIUM RISK", C["orange"], "⚠️"
    else:
        risk_label, risk_color, risk_emoji = "HIGH RISK",   C["red"],    "🚨"

    dec_color = C["red"] if decision=="DEFAULT" else C["green"]

    output_card = dbc.Card([dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H2(f"{risk_emoji}  {prob:.1%}",
                        style={"color":risk_color,"fontSize":"3rem"}),
                html.H4(risk_label, style={"color":risk_color}),
            ], md=7),
            dbc.Col([
                html.H3(decision, style={"color":dec_color,"fontWeight":"bold","marginTop":"10px"}),
                html.Small(f"Threshold = {round(threshold,2)}", className="text-muted"),
            ], md=5),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([html.Small("Grade",    className="text-muted"), html.H5(grade)],        md=3),
            dbc.Col([html.Small("Int Rate", className="text-muted"), html.H5(f"{int_rate}%")], md=3),
            dbc.Col([html.Small("FICO",     className="text-muted"), html.H5(fico)],          md=3),
            dbc.Col([html.Small("DTI",      className="text-muted"), html.H5(dti)],           md=3),
        ]),
        html.Hr(),
        html.Small(
            f"Predicted by XGBoost trained on {len(df):,} Lending Club loans (2007–2018). "
            f"Missing inputs filled with training set medians. "
            f"Optimal threshold (max F1): {OPTIMAL_THRESHOLD:.2f}.",
            className="text-muted"),
    ])], style={"borderLeft":f"6px solid {risk_color}"})

    # Distribution context
    ctx_pairs = [
        ("fico_avg",   fico,      "FICO Score"),
        ("int_rate",   int_rate,  "Interest Rate (%)"),
        ("dti",        dti,       "DTI Ratio"),
    ]
    fig_ctx = make_subplots(
        rows=1, cols=3,
        subplot_titles=[p[2] for p in ctx_pairs],
        horizontal_spacing=0.10,
    )
    for col_idx, (col_name, user_val, col_label) in enumerate(ctx_pairs, start=1):
        cap  = df[col_name].quantile(0.99)
        data = df[df[col_name] <= cap]
        for lbl, color in [("No Default", C["green"]), ("Default", C["red"])]:
            subset = data[data["default_label"] == lbl][col_name]
            fig_ctx.add_trace(go.Histogram(
                x=subset, name=lbl, opacity=0.55, marker_color=color,
                showlegend=(col_idx == 1), nbinsx=45,
                histnorm="probability density"),
                row=1, col=col_idx)
        fig_ctx.add_vline(
            x=user_val, line_dash="dash", line_color="#2c3e50", line_width=2,
            row=1, col=col_idx,
            annotation_text=f"You: {user_val}",
            annotation_font=dict(size=11, color="#2c3e50"),
            annotation_position="top right")
        fig_ctx.update_xaxes(title_text=col_label, row=1, col=col_idx,
                              title_font=dict(size=11))
    fig_ctx.update_layout(
        barmode="overlay",
        title=dict(
            text="Where does this borrower sit in the dataset?",
            font=dict(size=14),
            y=0.97, x=0, xanchor="left"),
        plot_bgcolor=C["bg"],
        paper_bgcolor="white",
        height=420,
        margin=dict(t=110, b=50, l=40, r=20),
        legend=dict(
            orientation="h",
            yanchor="top", y=1.0,
            xanchor="right", x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#dfe6e9", borderwidth=1,
            font=dict(size=11)))
    # push subplot titles down so legend does not overlap them
    for ann in fig_ctx.layout.annotations:
        if ann.text in [p[2] for p in ctx_pairs]:
            ann.y = ann.y - 0.04

    # Mini SHAP waterfall for this specific input
    sv = SHAP_EXPLAINER(X_input)
    plt.figure(figsize=(10,5))
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    plt.title("What drove this prediction?", fontweight="bold")
    plt.tight_layout()
    shap_img = _fig_to_b64()

    return output_card, fig_ctx, shap_img


# ─────────────────────────────────────────────────────────────────
# 11. RUN
# ─────────────────────────────────────────────────────────────────
# Expose Flask server for gunicorn (required for Render deployment)
server = app.server

if __name__ == "__main__":
    app.run(debug=False, port=8050)
