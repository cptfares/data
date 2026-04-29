# -*- coding: utf-8 -*-
"""
Resume–Job Matching · Dynamic Dash Dashboard
  Tab 1 – existing dataset overview
  Tab 2 – new dataset pipeline wizard (Upload → Discover → Preprocess → Train)
Run:  python dashboard.py   →   http://127.0.0.1:8050
"""

import re, ast, warnings, base64, io
import numpy as np
import pandas as pd
from collections import Counter
from io import StringIO

import dash
from dash import dcc, html, Input, Output, State, no_update, ctx, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# server-side model store (single-user local app)
_model_store: dict = {"model": None, "features": [], "target": None}

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
PRIMARY   = "#2c7be5"
SECONDARY = "#6e84a3"
SUCCESS   = "#00d97e"
WARNING   = "#f6c343"
DANGER    = "#e63757"
LIGHT_BG  = "#f5f7fb"
CARD_BG   = "#ffffff"

CARD = {
    "borderRadius": "12px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
    "background": CARD_BG,
    "padding": "16px",
    "marginBottom": "20px",
}

BTN = {"borderRadius": "8px", "fontWeight": "600", "fontSize": "14px"}

# ═══════════════════════════════════════════════════════════
# TAB-1  DATA LOADING & CLEANING PIPELINE
# ═══════════════════════════════════════════════════════════
FILE_PATH = "resume_data.csv"

NULL_TOKENS = {
    "unknown", "n/a", "na", "none", "null", "", " ",
    "not specified", "[none]", "['n/a']", "[n/a]",
    "city, state", "city , state",
}
JUNK_VALUES = {"none", "nan", "n/a", "unknown", "", " ", "not specified"}


def is_null_value(val):
    if pd.isna(val): return True
    return str(val).strip().lower() in NULL_TOKENS


def to_clean_list(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return []
    temp_s = ",".join(str(i).replace("\n", ",") for i in val) if isinstance(val, list) \
             else str(val).strip().replace("\n", ",")
    if temp_s.lower() in JUNK_VALUES or not temp_s.strip(): return []
    try:
        parsed = ast.literal_eval(temp_s)
        if isinstance(parsed, (list, tuple)):
            items = []
            for item in parsed:
                items.extend([i.strip() for i in str(item).split(",")
                               if i.strip().lower() not in JUNK_VALUES])
            return [i for i in items if i]
        return [str(parsed).strip()] if str(parsed).strip().lower() not in JUNK_VALUES else []
    except (ValueError, SyntaxError): pass
    return [i.strip() for i in temp_s.split(",")
            if i.strip().lower() not in JUNK_VALUES and i.strip()]


TODAY = pd.Timestamp.now()
PRESENT_TOKENS = {"present","till date","current","till now","ongoing","date","now"}


def parse_date_str(s):
    if pd.isna(s): return pd.NaT
    s = str(s).strip().lower()
    if s in PRESENT_TOKENS: return TODAY
    if s in ("","n/a","none","nan","unknown"): return pd.NaT
    if re.fullmatch(r"\d{4}", s): return pd.Timestamp(f"{s}-01-01")
    try:
        ts = pd.to_datetime(s, errors="coerce")
        return ts if pd.notna(ts) else pd.NaT
    except Exception: return pd.NaT


def parse_date_col(val):
    tokens = val if isinstance(val, list) else \
             [] if pd.isna(val) else [t.strip() for t in str(val).split(",")]
    return [r for r in (parse_date_str(t) for t in tokens) if pd.notna(r)]


def total_experience(starts, ends):
    return round(sum(
        (e - s).days / 365.25
        for s, e in zip(starts, ends)
        if pd.notna(s) and pd.notna(e) and e > s
    ), 2)


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = (df.columns.str.replace("﻿","",regex=False)
                  .str.replace("ï»¿","",regex=False).str.strip().str.lower())
    df.drop(columns=["educational_results","age_requirement","proficiency_levels",
                     "locations","result_types","company_urls","online_links",
                     "extra_curricular_organization_links"], errors="ignore", inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            mask = df[col].apply(is_null_value)
            df.loc[mask, col] = np.nan
    for col in ["address","career_objective","languages",
                "extra_curricular_activity_types","certification_providers"]:
        if col in df.columns: df[f"has_{col}"] = df[col].notna().astype(int)
    df.drop(columns=["extra_curricular_organization_names","role_positions",
                     "has_role_positions","certification_skills",
                     "issue_dates","expiry_dates"], errors="ignore", inplace=True)
    list_cols = ["skills","degree_names","educational_institution_name",
                 "professional_company_names","positions","related_skils_in_job",
                 "major_field_of_studies","educational_requirements",
                 "experiencere_requirement","skills_required",
                 "certification_providers","languages"]
    for col in list_cols:
        if col in df.columns: df[col] = df[col].apply(to_clean_list)
    for col in ["passing_years","start_dates","end_dates"]:
        if col in df.columns: df[col+"_parsed"] = df[col].apply(parse_date_col)
    df["total_years_experience"] = df.apply(
        lambda r: total_experience(r.get("start_dates_parsed",[]),
                                   r.get("end_dates_parsed",[])), axis=1)
    df.drop(columns=["passing_years","start_dates","end_dates",
                     "passing_years_parsed","start_dates_parsed","end_dates_parsed"],
            errors="ignore", inplace=True)
    def clean_text(v):
        if pd.isna(v): return ""
        return re.sub(r"\s+"," ", re.sub(r"ï\S{1,4}","",str(v))).strip()
    for col in ["career_objective","responsibilities","responsibilities.1","address","job_position_name"]:
        if col in df.columns: df[col] = df[col].apply(clean_text)
    if "matched_score" in df.columns: df["matched_score"] = df["matched_score"].clip(0,1)
    df["total_years_experience"] = df["total_years_experience"].clip(0,40)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x,list)).any():
            df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x,list) else x)
    df.drop_duplicates(inplace=True)
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: list(x) if isinstance(x,tuple) else x)
    df["has_valid_experience"] = (df["total_years_experience"] > 0).astype(int)
    df["num_skills"]         = df["skills"].apply(lambda x: len(x) if isinstance(x,list) else 0)
    df["num_positions"]      = df["positions"].apply(lambda x: len(x) if isinstance(x,list) else 0)
    df["num_related_skills"] = df["related_skils_in_job"].apply(lambda x: len(x) if isinstance(x,list) else 0)
    df["num_degrees"]        = df["degree_names"].apply(lambda x: len(x) if isinstance(x,list) else 0)
    return df


print("Loading data …", end=" ", flush=True)
df = load_and_clean(FILE_PATH)
print(f"done — {df.shape[0]:,} rows × {df.shape[1]} cols")

all_skills      = [s for row in df["skills"] if isinstance(row,list) for s in row]
skill_counts    = Counter(all_skills)
all_req_skills  = [s for row in df["skills_required"] if isinstance(row,list) for s in row]
req_skill_counts = Counter(all_req_skills)
num_cols_t1     = df.select_dtypes(include=np.number).columns.tolist()
avg_score   = df["matched_score"].mean()  if "matched_score" in df.columns else 0
high_match  = int((df["matched_score"]>=0.7).sum()) if "matched_score" in df.columns else 0
unique_jobs = df["job_position_name"].nunique() if "job_position_name" in df.columns else 0

# ═══════════════════════════════════════════════════════════
# TAB-2  PIPELINE HELPERS
# ═══════════════════════════════════════════════════════════

def parse_upload(contents, filename):
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(StringIO(decoded.decode("utf-8")))
    return pd.read_excel(io.BytesIO(decoded))


def df_to_store(df_in):
    d = df_in.copy()
    for col in d.columns:
        if d[col].apply(lambda x: isinstance(x,(list,tuple))).any():
            d[col] = d[col].apply(lambda x: "; ".join(str(i) for i in x)
                                  if isinstance(x,(list,tuple)) else x)
    return d.to_json(orient="split", date_format="iso")


def store_to_df(data): return pd.read_json(StringIO(data), orient="split")


def generic_preprocess(df_in, null_thresh, fill_num, fill_cat, remove_dupes):
    df2 = df_in.copy()
    df2.columns = (df2.columns.str.strip().str.lower()
                   .str.replace(r"\s+","_",regex=True)
                   .str.replace(r"[^\w]","",regex=True))
    orig = df2.shape
    null_pct  = df2.isnull().mean()
    drop_cols = null_pct[null_pct > null_thresh/100].index.tolist()
    df2.drop(columns=drop_cols, inplace=True)
    ncols = df2.select_dtypes(include=np.number).columns
    ccols = df2.select_dtypes(exclude=np.number).columns
    if fill_num == "median": df2[ncols] = df2[ncols].fillna(df2[ncols].median())
    elif fill_num == "mean": df2[ncols] = df2[ncols].fillna(df2[ncols].mean())
    elif fill_num == "zero":  df2[ncols] = df2[ncols].fillna(0)
    if fill_cat == "mode":
        for c in ccols:
            m = df2[c].mode(); df2[c] = df2[c].fillna(m[0] if len(m) else "")
    elif fill_cat == "empty": df2[ccols] = df2[ccols].fillna("")
    rows_before = len(df2)
    if remove_dupes: df2.drop_duplicates(inplace=True)
    return df2, {"orig": orig, "new": df2.shape,
                 "dropped_cols": drop_cols, "rows_dropped": rows_before - len(df2)}


def train_models(df_in, target, model_names, test_size_pct):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from xgboost import XGBRegressor
    ncols = df_in.select_dtypes(include=np.number).columns.tolist()
    feats = [c for c in ncols if c != target and df_in[c].std() > 0]
    X = df_in[feats].fillna(0)
    y = df_in[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size_pct/100, random_state=42)
    catalogue = {
        "XGBoost":           XGBRegressor(n_estimators=150, learning_rate=0.05,
                                          max_depth=4, random_state=42, n_jobs=-1,
                                          verbosity=0),
        "Random Forest":     RandomForestRegressor(n_estimators=150, max_depth=6,
                                                   random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                       max_depth=3, random_state=42),
    }
    results, best_r2, best_name, best_preds, importances, best_obj = {}, -999, None, None, {}, None
    for name in model_names:
        if name not in catalogue: continue
        m = catalogue[name]; m.fit(X_tr, y_tr); p = m.predict(X_te)
        mae = mean_absolute_error(y_te, p); r2 = r2_score(y_te, p)
        results[name] = {"MAE": round(mae,4), "R²": round(r2,4)}
        if r2 > best_r2:
            best_r2, best_name, best_preds, best_obj = r2, name, p.tolist(), m
            if hasattr(m,"feature_importances_"):
                importances = dict(zip(feats, m.feature_importances_.tolist()))
    return {"results": results, "best": best_name,
            "y_test": y_te.tolist(), "preds": best_preds,
            "importances": importances, "feats": feats,
            "model_obj": best_obj}


# ═══════════════════════════════════════════════════════════
# LAYOUT HELPERS
# ═══════════════════════════════════════════════════════════

def kpi_card(title, value, color=PRIMARY):
    return dbc.Col(html.Div([
        html.P(title, style={"color":SECONDARY,"fontSize":"12px","marginBottom":"4px",
                             "fontWeight":"700","textTransform":"uppercase","letterSpacing":"0.05em"}),
        html.H3(str(value), style={"color":color,"fontWeight":"800","margin":0}),
    ], style={**CARD,"textAlign":"center","paddingTop":"20px","paddingBottom":"20px"}),
    xs=12, sm=6, md=3)


def step_indicator(active):
    labels = ["Upload","Discover","Preprocess","Train"]
    items  = []
    for i, label in enumerate(labels, 1):
        done   = i < active
        cur    = i == active
        circle_bg  = SUCCESS if done else (PRIMARY if cur else "#dee2e6")
        circle_col = "#fff"  if (done or cur) else SECONDARY
        symbol     = "✓"    if done else str(i)
        items.append(html.Div([
            html.Div(symbol, style={
                "width":"36px","height":"36px","borderRadius":"50%",
                "background":circle_bg,"color":circle_col,
                "display":"flex","alignItems":"center","justifyContent":"center",
                "fontWeight":"700","fontSize":"16px","margin":"0 auto","transition":"all .3s"}),
            html.P(label, style={"fontSize":"12px","color": PRIMARY if cur else SECONDARY,
                                 "fontWeight":"700" if cur else "400",
                                 "marginTop":"6px","marginBottom":0,"textAlign":"center"}),
        ], style={"flex":"0 0 auto","width":"70px"}))
        if i < len(labels):
            items.append(html.Div(style={
                "flex":"1","height":"2px",
                "background": SUCCESS if i < active else "#dee2e6",
                "marginTop":"18px","transition":"all .3s",
            }))
    return html.Div(items, style={
        "display":"flex","alignItems":"flex-start",
        "padding":"20px 40px 10px","background":CARD_BG,
        "borderRadius":"12px","marginBottom":"20px",
        "boxShadow":"0 2px 8px rgba(0,0,0,0.08)",
    })


# ═══════════════════════════════════════════════════════════
# APP & LAYOUT
# ═══════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)
app.title = "Resume Matching Dashboard"

# ── Tab-1 content ───────────────────────────────────────────
tab1_content = dbc.Container(fluid=True, style={"background":LIGHT_BG,"padding":"30px"}, children=[
    dbc.Row([dbc.Col(html.Div([
        html.H4("Resume–Job Matching Dashboard",
                style={"fontWeight":"800","color":"#12263f","margin":0}),
        html.P(f"{df.shape[0]:,} resumes · {df.shape[1]} features",
               style={"color":SECONDARY,"fontSize":"13px","marginTop":"4px"}),
    ]))], className="mb-4"),
    dbc.Row([
        kpi_card("Total Resumes", f"{len(df):,}",  PRIMARY),
        kpi_card("Avg Match Score", f"{avg_score:.3f}", SUCCESS),
        kpi_card("High Match ≥0.7", f"{high_match:,}", WARNING),
        kpi_card("Unique Job Roles", f"{unique_jobs}",  DANGER),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(html.Div([
            html.H6("Match Score Distribution", style={"fontWeight":"700"}),
            dcc.RangeSlider(id="score-range", min=0, max=1, step=0.01, value=[0,1],
                            marks={i/10:f"{i/10:.1f}" for i in range(0,11)},
                            tooltip={"placement":"bottom"}),
            dcc.Graph(id="score-hist", style={"height":"300px"}),
        ], style=CARD), md=6),
        dbc.Col(html.Div([
            html.H6("Top Job Positions", style={"fontWeight":"700"}),
            dcc.Slider(id="top-n-jobs", min=5, max=30, step=5, value=15,
                       marks={n:str(n) for n in range(5,35,5)},
                       tooltip={"placement":"bottom"}),
            dcc.Graph(id="job-bar", style={"height":"300px"}),
        ], style=CARD), md=6),
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            html.H6("Top Resume Skills", style={"fontWeight":"700"}),
            dcc.Slider(id="top-n-skills", min=5, max=30, step=5, value=15,
                       marks={n:str(n) for n in range(5,35,5)},
                       tooltip={"placement":"bottom"}),
            dcc.Graph(id="skills-bar", style={"height":"300px"}),
        ], style=CARD), md=6),
        dbc.Col(html.Div([
            html.H6("Top Required Skills (Job Descriptions)", style={"fontWeight":"700"}),
            dcc.Slider(id="top-n-req", min=5, max=30, step=5, value=15,
                       marks={n:str(n) for n in range(5,35,5)},
                       tooltip={"placement":"bottom"}),
            dcc.Graph(id="req-skills-bar", style={"height":"300px"}),
        ], style=CARD), md=6),
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            html.H6("Feature Correlation with Match Score", style={"fontWeight":"700"}),
            dcc.Graph(id="corr-bar", style={"height":"340px"}),
        ], style=CARD), md=6),
        dbc.Col(html.Div([
            html.H6("Resume Field Completeness", style={"fontWeight":"700"}),
            dcc.Graph(id="completeness-bar", style={"height":"340px"}),
        ], style=CARD), md=6),
    ]),
    dbc.Row([dbc.Col(html.Div([
        html.H6("Scatter Explorer", style={"fontWeight":"700"}),
        dbc.Row([
            dbc.Col([html.Label("X-axis", style={"fontSize":"12px"}),
                     dcc.Dropdown(id="scatter-x",
                                  options=[{"label":c,"value":c} for c in num_cols_t1 if c!="matched_score"],
                                  value="num_skills" if "num_skills" in num_cols_t1 else None,
                                  clearable=False)], md=6),
            dbc.Col([html.Label("Colour", style={"fontSize":"12px"}),
                     dcc.Dropdown(id="scatter-color",
                                  options=[{"label":"None","value":"none"},
                                           {"label":"Job position","value":"job_position_name"}],
                                  value="none", clearable=False)], md=6),
        ], className="mb-2"),
        dcc.Graph(id="scatter-plot", style={"height":"360px"}),
    ], style=CARD))]),
    dbc.Row([dbc.Col(html.Div([
        html.H6("Explore Filtered Resumes", style={"fontWeight":"700"}),
        dbc.Row([
            dbc.Col([html.Label("Min Match Score", style={"fontSize":"12px"}),
                     dcc.Slider(id="table-min-score", min=0, max=1, step=0.01, value=0,
                                marks={i/10:f"{i/10:.1f}" for i in range(0,11)},
                                tooltip={"placement":"bottom"})], md=6),
            dbc.Col([html.Label("Job Position", style={"fontSize":"12px"}),
                     dcc.Dropdown(id="table-job-filter",
                                  options=[{"label":"All","value":"all"}]+[
                                      {"label":j,"value":j} for j in
                                      sorted(df["job_position_name"].dropna().unique())],
                                  value="all", clearable=False)], md=6),
        ], className="mb-3"),
        html.Div(id="table-count", style={"color":SECONDARY,"fontSize":"13px","marginBottom":"8px"}),
        dash_table.DataTable(id="resume-table", page_size=10, page_action="native",
                             sort_action="native", filter_action="native",
                             style_table={"overflowX":"auto"},
                             style_header={"backgroundColor":PRIMARY,"color":"white",
                                           "fontWeight":"700","fontSize":"12px"},
                             style_cell={"fontSize":"12px","padding":"8px 12px",
                                         "maxWidth":"180px","overflow":"hidden","textOverflow":"ellipsis"},
                             style_data_conditional=[{"if":{"row_index":"odd"},
                                                      "backgroundColor":"#f8f9fa"}]),
    ], style=CARD))]),
])

# ── Tab-2 content ───────────────────────────────────────────
def _panel(step_id, children):
    return html.Div(children, id=step_id, style={"display":"none"})


tab2_content = dbc.Container(fluid=True, style={"background":LIGHT_BG,"padding":"30px"}, children=[

    # persistent stores
    dcc.Store(id="store-step",      data=1),
    dcc.Store(id="store-raw"),
    dcc.Store(id="store-cleaned"),
    dcc.Store(id="store-train"),
    dcc.Store(id="store-prep-done", data=False),
    dcc.Store(id="store-model-meta"),       # feature list + target used for training
    dcc.Download(id="download-predictions"),

    html.H4("New Dataset Pipeline", style={"fontWeight":"800","color":"#12263f",
                                            "marginBottom":"20px"}),
    html.Div(id="stepper-area"),

    # ── Step 1: Upload ──────────────────────────────────────
    _panel("panel-1", [
        html.Div([
            html.H6("Step 1 · Upload Dataset", style={"fontWeight":"700","marginBottom":"16px"}),
            dcc.Upload(id="upload-data",
                children=html.Div([
                    html.Div("📂", style={"fontSize":"48px","marginBottom":"12px"}),
                    html.P("Drag & Drop  or  ", style={"display":"inline","color":SECONDARY}),
                    html.A("Browse File", style={"color":PRIMARY,"fontWeight":"600","cursor":"pointer"}),
                    html.P("Supports CSV · Excel", style={"color":"#aaa","fontSize":"12px",
                                                          "marginTop":"8px"}),
                ], style={"textAlign":"center","padding":"30px"}),
                style={"border":f"2px dashed {PRIMARY}","borderRadius":"12px",
                       "cursor":"pointer","background":"#f0f6ff"},
                multiple=False),
            html.Div(id="upload-info", style={"marginTop":"16px"}),
            html.Div(
                dbc.Button("Run Discovery →", id="btn-to-2", color="primary",
                           disabled=True, style=BTN),
                style={"marginTop":"20px","textAlign":"right"}),
        ], style=CARD),
    ]),

    # ── Step 2: Discovery ───────────────────────────────────
    _panel("panel-2", [
        html.Div([
            html.H6("Step 2 · Data Discovery", style={"fontWeight":"700","marginBottom":"16px"}),
            html.Div(id="discovery-content"),
            dbc.Row([
                dbc.Col(dbc.Button("← Back", id="btn-back-2", outline=True,
                                   color="secondary", style=BTN), width="auto"),
                dbc.Col(dbc.Button("Proceed to Preprocessing →", id="btn-to-3",
                                   color="primary", style=BTN), width="auto"),
            ], justify="between", style={"marginTop":"20px"}),
        ], style=CARD),
    ]),

    # ── Step 3: Preprocessing ───────────────────────────────
    _panel("panel-3", [
        html.Div([
            html.H6("Step 3 · Preprocessing", style={"fontWeight":"700","marginBottom":"16px"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Drop columns with missing values above (%)",
                               style={"fontSize":"13px","fontWeight":"600"}),
                    dcc.Slider(id="prep-null-thresh", min=20, max=100, step=10, value=80,
                               marks={v:f"{v}%" for v in range(20,110,20)},
                               tooltip={"placement":"bottom"}),
                ], md=12, className="mb-3"),
                dbc.Col([
                    html.Label("Fill missing numeric values", style={"fontSize":"13px","fontWeight":"600"}),
                    dcc.RadioItems(id="prep-fill-num",
                                   options=[{"label":" Median","value":"median"},
                                            {"label":" Mean","value":"mean"},
                                            {"label":" Zero","value":"zero"},
                                            {"label":" Leave as-is","value":"none"}],
                                   value="median", inline=True,
                                   labelStyle={"marginRight":"18px"}),
                ], md=6, className="mb-3"),
                dbc.Col([
                    html.Label("Fill missing text values", style={"fontSize":"13px","fontWeight":"600"}),
                    dcc.RadioItems(id="prep-fill-cat",
                                   options=[{"label":" Mode","value":"mode"},
                                            {"label":" Empty string","value":"empty"},
                                            {"label":" Leave as-is","value":"none"}],
                                   value="mode", inline=True,
                                   labelStyle={"marginRight":"18px"}),
                ], md=6, className="mb-3"),
                dbc.Col([
                    dcc.Checklist(id="prep-dedup",
                                  options=[{"label":"  Remove duplicate rows","value":"yes"}],
                                  value=["yes"],
                                  labelStyle={"fontWeight":"600","fontSize":"13px"}),
                ], md=12, className="mb-3"),
            ]),
            dbc.Button("⚙️  Run Preprocessing", id="btn-run-prep", color="warning", style=BTN),
            dcc.Loading(html.Div(id="prep-results", style={"marginTop":"20px"}),
                        type="circle", color=PRIMARY),
            dbc.Row([
                dbc.Col(dbc.Button("← Back", id="btn-back-3", outline=True,
                                   color="secondary", style=BTN), width="auto"),
                dbc.Col(dbc.Button("Proceed to Training →", id="btn-to-4",
                                   color="primary", disabled=True, style=BTN), width="auto"),
            ], justify="between", style={"marginTop":"20px"}),
        ], style=CARD),
    ]),

    # ── Step 4: Train / Predict ─────────────────────────────
    _panel("panel-4", [
        html.Div([
            html.H6("Step 4 · Train or Predict", style={"fontWeight":"700","marginBottom":"16px"}),

            # mode toggle
            dbc.Row([
                dbc.Col(
                    dcc.RadioItems(
                        id="step4-mode",
                        options=[
                            {"label": "  🚀  Train a new model",        "value": "train"},
                            {"label": "  🔮  Predict on this dataset",   "value": "predict"},
                        ],
                        value="train", inline=True,
                        inputStyle={"marginRight":"6px"},
                        labelStyle={"marginRight":"28px","fontWeight":"600","fontSize":"14px"},
                    ), md=12, className="mb-4",
                ),
            ]),

            # ── TRAIN SECTION ──────────────────────────────────
            html.Div(id="section-train", children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Target column", style={"fontSize":"13px","fontWeight":"600"}),
                        dcc.Dropdown(id="train-target", clearable=False, style={"fontSize":"13px"}),
                    ], md=4, className="mb-3"),
                    dbc.Col([
                        html.Label("Models to train", style={"fontSize":"13px","fontWeight":"600"}),
                        dcc.Checklist(id="train-models",
                                      options=[{"label":"  XGBoost",           "value":"XGBoost"},
                                               {"label":"  Random Forest",     "value":"Random Forest"},
                                               {"label":"  Gradient Boosting", "value":"Gradient Boosting"}],
                                      value=["XGBoost","Random Forest"],
                                      labelStyle={"marginRight":"18px","fontSize":"13px"}),
                    ], md=4, className="mb-3"),
                    dbc.Col([
                        html.Label("Test size (%)", style={"fontSize":"13px","fontWeight":"600"}),
                        dcc.Slider(id="train-test-size", min=10, max=40, step=5, value=20,
                                   marks={v:f"{v}%" for v in range(10,45,10)},
                                   tooltip={"placement":"bottom"}),
                    ], md=4, className="mb-3"),
                ]),
                dbc.Button("🚀  Run Training", id="btn-run-train", color="success", style=BTN),
                dcc.Loading(html.Div(id="train-results", style={"marginTop":"20px"}),
                            type="circle", color=SUCCESS),
            ]),

            # ── PREDICT SECTION ────────────────────────────────
            html.Div(id="section-predict", style={"display":"none"}, children=[
                html.Div(id="predict-model-status", style={"marginBottom":"16px"}),
                dbc.Row([
                    dbc.Col([
                        html.Label("Compare predictions against (optional)",
                                   style={"fontSize":"13px","fontWeight":"600"}),
                        dcc.Dropdown(id="predict-actual-col", placeholder="— skip —",
                                     clearable=True, style={"fontSize":"13px"}),
                    ], md=5, className="mb-3"),
                ]),
                dbc.Button("🔮  Run Prediction", id="btn-run-predict", color="primary", style=BTN),
                html.Span("  "),
                dbc.Button("⬇️  Download CSV", id="btn-download-pred",
                           color="secondary", outline=True, style={**BTN,"marginLeft":"8px"},
                           disabled=True),
                dcc.Loading(html.Div(id="predict-results", style={"marginTop":"20px"}),
                            type="circle", color=PRIMARY),
            ]),

            dbc.Row([
                dbc.Col(dbc.Button("← Back", id="btn-back-4", outline=True,
                                   color="secondary", style=BTN), width="auto"),
            ], style={"marginTop":"24px"}),
        ], style=CARD),
    ]),
])

# ── Root layout ─────────────────────────────────────────────
app.layout = html.Div([
    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label="📊  Dashboard", value="tab-1",
                style={"fontWeight":"600"}, selected_style={"fontWeight":"800","color":PRIMARY}),
        dcc.Tab(label="🔬  New Dataset Pipeline", value="tab-2",
                style={"fontWeight":"600"}, selected_style={"fontWeight":"800","color":PRIMARY}),
    ], style={"marginBottom":0,"borderBottom":f"3px solid {PRIMARY}"}),
    html.Div(id="tab-content"),
], style={"background":LIGHT_BG,"minHeight":"100vh"})


# ═══════════════════════════════════════════════════════════
# TAB ROUTER
# ═══════════════════════════════════════════════════════════
@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):
    return tab1_content if tab=="tab-1" else tab2_content


# ═══════════════════════════════════════════════════════════
# TAB-1 CALLBACKS
# ═══════════════════════════════════════════════════════════

@app.callback(Output("score-hist","figure"), Input("score-range","value"))
def score_hist(rng):
    lo,hi = rng
    filt = df[(df["matched_score"]>=lo)&(df["matched_score"]<=hi)]
    fig = px.histogram(filt, x="matched_score", nbins=40, color_discrete_sequence=[PRIMARY])
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=30),
                      plot_bgcolor="white",paper_bgcolor="white",bargap=0.05)
    fig.add_vline(x=filt["matched_score"].mean(), line_dash="dash", line_color=DANGER,
                  annotation_text=f"mean={filt['matched_score'].mean():.3f}",
                  annotation_position="top right")
    return fig


@app.callback(Output("job-bar","figure"), Input("top-n-jobs","value"))
def job_bar(n):
    top = df["job_position_name"].value_counts().head(n).reset_index()
    top.columns = ["job","count"]
    fig = px.bar(top.sort_values("count"), x="count", y="job", orientation="h",
                 color="count", color_continuous_scale=[[0,"#c3d8f7"],[1,PRIMARY]],
                 labels={"count":"# Resumes","job":""})
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=20),
                      plot_bgcolor="white",paper_bgcolor="white",coloraxis_showscale=False)
    return fig


@app.callback(Output("skills-bar","figure"), Input("top-n-skills","value"))
def skills_bar(n):
    top = pd.DataFrame(skill_counts.most_common(n), columns=["skill","count"])
    fig = px.bar(top.sort_values("count"), x="count", y="skill", orientation="h",
                 color="count", color_continuous_scale=[[0,"#c3f7e0"],[1,SUCCESS]],
                 labels={"count":"Freq","skill":""})
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=20),
                      plot_bgcolor="white",paper_bgcolor="white",coloraxis_showscale=False)
    return fig


@app.callback(Output("req-skills-bar","figure"), Input("top-n-req","value"))
def req_skills_bar(n):
    top = pd.DataFrame(req_skill_counts.most_common(n), columns=["skill","count"])
    fig = px.bar(top.sort_values("count"), x="count", y="skill", orientation="h",
                 color="count", color_continuous_scale=[[0,"#fce8c3"],[1,WARNING]],
                 labels={"count":"Freq","skill":""})
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=20),
                      plot_bgcolor="white",paper_bgcolor="white",coloraxis_showscale=False)
    return fig


@app.callback(Output("corr-bar","figure"), Input("score-range","value"))
def corr_bar(rng):
    lo,hi = rng
    filt = df[(df["matched_score"]>=lo)&(df["matched_score"]<=hi)]
    nc = filt.select_dtypes(include=np.number).columns.tolist()
    if "matched_score" not in nc or len(nc)<2: return go.Figure()
    c = filt[nc].corr()["matched_score"].drop("matched_score").sort_values()
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h",
                           marker_color=[DANGER if v<0 else PRIMARY for v in c.values]))
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=20),
                      plot_bgcolor="white",paper_bgcolor="white",xaxis_title="Pearson r")
    fig.add_vline(x=0, line_color="black", line_width=0.8)
    return fig


@app.callback(Output("completeness-bar","figure"), Input("score-range","value"))
def completeness_bar(_):
    has_cols = [c for c in df.columns if c.startswith("has_")]
    if not has_cols: return go.Figure()
    means = df[has_cols].mean().sort_values()
    fig = go.Figure(go.Bar(x=means.values,
                           y=[c.replace("has_","").replace("_"," ") for c in means.index],
                           orientation="h",
                           marker_color=[SUCCESS if v>=0.5 else DANGER for v in means.values]))
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=20),
                      plot_bgcolor="white",paper_bgcolor="white",
                      xaxis_title="Proportion",xaxis_range=[0,1])
    fig.add_vline(x=0.5, line_dash="dash", line_color=SECONDARY, line_width=1)
    return fig


@app.callback(Output("scatter-plot","figure"),
              Input("scatter-x","value"), Input("scatter-color","value"),
              Input("score-range","value"))
def scatter_plot(xcol, color_col, rng):
    if not xcol: return go.Figure()
    lo,hi = rng
    filt = df[(df["matched_score"]>=lo)&(df["matched_score"]<=hi)]
    sample = filt.sample(min(3000,len(filt)), random_state=42)
    color = None if color_col=="none" else color_col
    fig = px.scatter(sample, x=xcol, y="matched_score", color=color, opacity=0.45)
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=20),
                      plot_bgcolor="white",paper_bgcolor="white")
    return fig


T1_COLS = [c for c in ["job_position_name","matched_score","total_years_experience",
                        "num_skills","num_positions","num_degrees",
                        "has_address","has_career_objective","has_certification_providers"]
           if c in df.columns]


@app.callback(Output("resume-table","data"), Output("resume-table","columns"),
              Output("table-count","children"),
              Input("table-min-score","value"), Input("table-job-filter","value"))
def update_table(min_s, job):
    filt = df[df["matched_score"]>=min_s] if "matched_score" in df.columns else df
    if job and job!="all" and "job_position_name" in filt.columns:
        filt = filt[filt["job_position_name"]==job]
    out = filt[T1_COLS].copy()
    for c in out.select_dtypes(include=np.number).columns: out[c] = out[c].round(4)
    cols = [{"name":c.replace("_"," ").title(),"id":c} for c in T1_COLS]
    return out.to_dict("records"), cols, f"Showing {len(out):,} resumes"


# ═══════════════════════════════════════════════════════════
# TAB-2 CALLBACKS
# ═══════════════════════════════════════════════════════════

# ── Step navigation ─────────────────────────────────────────
@app.callback(
    Output("store-step","data"),
    Input("btn-to-2","n_clicks"),
    Input("btn-to-3","n_clicks"),
    Input("btn-to-4","n_clicks"),
    Input("btn-back-2","n_clicks"),
    Input("btn-back-3","n_clicks"),
    Input("btn-back-4","n_clicks"),
    prevent_initial_call=True,
)
def navigate(*_):
    t = ctx.triggered_id
    if t=="btn-to-2":   return 2
    if t=="btn-to-3":   return 3
    if t=="btn-to-4":   return 4
    if t=="btn-back-2": return 1
    if t=="btn-back-3": return 2
    if t=="btn-back-4": return 3
    return 1


# ── Show/hide step panels + update stepper ──────────────────
@app.callback(
    Output("panel-1","style"), Output("panel-2","style"),
    Output("panel-3","style"), Output("panel-4","style"),
    Output("stepper-area","children"),
    Input("store-step","data"),
)
def show_panels(step):
    styles = [{"display":"none"}]*4
    styles[step-1] = {"display":"block"}
    return *styles, step_indicator(step)


# ── File upload ──────────────────────────────────────────────
@app.callback(
    Output("store-raw","data"),
    Output("upload-info","children"),
    Output("btn-to-2","disabled"),
    Input("upload-data","contents"),
    State("upload-data","filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if not contents:
        return no_update, no_update, True
    try:
        df_up = parse_upload(contents, filename)
    except Exception as e:
        return no_update, dbc.Alert(f"Error reading file: {e}", color="danger"), True

    rows, cols_ = df_up.shape
    info = html.Div([
        dbc.Alert([
            html.Strong(f"✅  {filename}"),
            html.Span(f"  ·  {rows:,} rows  ·  {cols_} columns",
                      style={"color":SECONDARY,"marginLeft":"8px"}),
        ], color="success", style={"marginBottom":"12px","padding":"10px 16px"}),
        html.P("Preview (first 5 rows):", style={"fontWeight":"600","fontSize":"13px"}),
        dash_table.DataTable(
            data=df_up.head(5).astype(str).to_dict("records"),
            columns=[{"name":c,"id":c} for c in df_up.columns],
            style_table={"overflowX":"auto"},
            style_header={"backgroundColor":PRIMARY,"color":"white","fontSize":"11px","fontWeight":"700"},
            style_cell={"fontSize":"11px","padding":"6px 10px","maxWidth":"140px",
                        "overflow":"hidden","textOverflow":"ellipsis"},
        ),
    ])
    return df_to_store(df_up), info, False


# ── Discovery (triggered when step becomes 2) ───────────────
@app.callback(
    Output("discovery-content","children"),
    Input("store-step","data"),
    State("store-raw","data"),
    prevent_initial_call=True,
)
def run_discovery(step, raw):
    if step!=2 or not raw: return no_update
    df_d = store_to_df(raw)
    rows, cols_ = df_d.shape
    n_miss  = int(df_d.isnull().sum().sum())
    n_num   = int(df_d.select_dtypes(include=np.number).shape[1])
    n_cat   = cols_ - n_num

    # missing value chart
    miss_df = (df_d.isnull().mean()*100).rename_axis("Column").reset_index(name="Missing %")
    miss_df["Missing %"] = miss_df["Missing %"].round(1)
    miss_top = miss_df[miss_df["Missing %"] > 0].sort_values("Missing %", ascending=True).tail(20)
    if miss_top.empty:
        fig_miss2 = go.Figure()
        fig_miss2.add_annotation(text="No missing values — dataset is complete!",
                                 xref="paper", yref="paper", x=0.5, y=0.5,
                                 showarrow=False, font=dict(size=14, color=SUCCESS))
        fig_miss2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                margin=dict(l=20,r=20,t=10,b=20))
    else:
        fig_miss2 = px.bar(miss_top, x="Missing %", y="Column", orientation="h",
                           color="Missing %",
                           color_continuous_scale=[[0,"#ffe0e0"],[1,DANGER]])
        fig_miss2.update_layout(margin=dict(l=20,r=20,t=10,b=20),
                                plot_bgcolor="white", paper_bgcolor="white",
                                coloraxis_showscale=False)

    # dtype breakdown
    dtype_counts = df_d.dtypes.apply(lambda x: "Numeric" if pd.api.types.is_numeric_dtype(x) else "Text/Other").value_counts()
    fig_types = px.pie(values=dtype_counts.values, names=dtype_counts.index,
                       color_discrete_sequence=[PRIMARY, WARNING],
                       hole=0.55)
    fig_types.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                            paper_bgcolor="white",showlegend=True)

    # numeric distribution summary
    num_df = df_d.select_dtypes(include=np.number)
    stats_html = None
    if not num_df.empty:
        desc = num_df.describe().T.round(3).reset_index().rename(columns={"index":"column"})
        stats_html = dash_table.DataTable(
            data=desc.to_dict("records"),
            columns=[{"name":c,"id":c} for c in desc.columns],
            style_table={"overflowX":"auto"},
            style_header={"backgroundColor":PRIMARY,"color":"white","fontSize":"11px","fontWeight":"700"},
            style_cell={"fontSize":"11px","padding":"6px 10px"},
            style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#f8f9fa"}],
        )

    return html.Div([
        # KPI row
        dbc.Row([
            dbc.Col(html.Div([html.P("Rows",style={"color":SECONDARY,"fontSize":"11px","fontWeight":"700","marginBottom":"2px","textTransform":"uppercase"}),
                              html.H4(f"{rows:,}",style={"color":PRIMARY,"margin":0,"fontWeight":"800"})],
                             style={**CARD,"textAlign":"center","paddingTop":"14px","paddingBottom":"14px"}), md=3),
            dbc.Col(html.Div([html.P("Columns",style={"color":SECONDARY,"fontSize":"11px","fontWeight":"700","marginBottom":"2px","textTransform":"uppercase"}),
                              html.H4(f"{cols_}",style={"color":PRIMARY,"margin":0,"fontWeight":"800"})],
                             style={**CARD,"textAlign":"center","paddingTop":"14px","paddingBottom":"14px"}), md=3),
            dbc.Col(html.Div([html.P("Missing Cells",style={"color":SECONDARY,"fontSize":"11px","fontWeight":"700","marginBottom":"2px","textTransform":"uppercase"}),
                              html.H4(f"{n_miss:,}",style={"color":DANGER if n_miss>0 else SUCCESS,"margin":0,"fontWeight":"800"})],
                             style={**CARD,"textAlign":"center","paddingTop":"14px","paddingBottom":"14px"}), md=3),
            dbc.Col(html.Div([html.P("Numeric / Text",style={"color":SECONDARY,"fontSize":"11px","fontWeight":"700","marginBottom":"2px","textTransform":"uppercase"}),
                              html.H4(f"{n_num} / {n_cat}",style={"color":SUCCESS,"margin":0,"fontWeight":"800"})],
                             style={**CARD,"textAlign":"center","paddingTop":"14px","paddingBottom":"14px"}), md=3),
        ], className="mb-3"),

        # charts
        dbc.Row([
            dbc.Col(html.Div([html.H6("Missing Values per Column",style={"fontWeight":"700"}),
                              dcc.Graph(figure=fig_miss2, style={"height":"300px"})], style=CARD), md=8),
            dbc.Col(html.Div([html.H6("Column Types",style={"fontWeight":"700"}),
                              dcc.Graph(figure=fig_types, style={"height":"300px"})], style=CARD), md=4),
        ]),

        # numeric stats
        html.Div([
            html.H6("Numeric Column Statistics", style={"fontWeight":"700"}),
            stats_html,
        ], style=CARD) if not num_df.empty else None,

        # sample
        html.Div([
            html.H6("Sample Data (first 10 rows)", style={"fontWeight":"700"}),
            dash_table.DataTable(
                data=df_d.head(10).astype(str).to_dict("records"),
                columns=[{"name":c,"id":c} for c in df_d.columns],
                style_table={"overflowX":"auto"},
                style_header={"backgroundColor":PRIMARY,"color":"white","fontSize":"11px","fontWeight":"700"},
                style_cell={"fontSize":"11px","padding":"6px 10px","maxWidth":"160px",
                            "overflow":"hidden","textOverflow":"ellipsis"},
                style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#f8f9fa"}],
                page_size=10,
            ),
        ], style=CARD),
    ])


# ── Preprocessing ────────────────────────────────────────────
@app.callback(
    Output("store-cleaned","data"),
    Output("prep-results","children"),
    Output("store-prep-done","data"),
    Output("btn-to-4","disabled"),
    Input("btn-run-prep","n_clicks"),
    State("store-raw","data"),
    State("prep-null-thresh","value"),
    State("prep-fill-num","value"),
    State("prep-fill-cat","value"),
    State("prep-dedup","value"),
    prevent_initial_call=True,
)
def run_preprocessing(n, raw, thresh, fill_num, fill_cat, dedup):
    if not raw: return no_update, dbc.Alert("Upload a file first.", color="warning"), False, True
    df_r = store_to_df(raw)
    remove_dupes = "yes" in (dedup or [])
    df_c, stats = generic_preprocess(df_r, thresh, fill_num, fill_cat, remove_dupes)

    orig_r, orig_c = stats["orig"]
    new_r,  new_c  = stats["new"]
    dropped = stats["dropped_cols"]

    results = html.Div([
        dbc.Alert("✅  Preprocessing complete!", color="success",
                  style={"padding":"10px 16px","marginBottom":"12px"}),
        dbc.Row([
            dbc.Col(html.Div([
                html.P("Before", style={"fontWeight":"700","color":SECONDARY,"fontSize":"12px","textTransform":"uppercase"}),
                html.H5(f"{orig_r:,} rows × {orig_c} cols", style={"fontWeight":"800","color":DANGER}),
            ], style={**CARD,"textAlign":"center","padding":"14px"}), md=5),
            dbc.Col(html.Div("→", style={"fontSize":"28px","textAlign":"center",
                                          "paddingTop":"18px","color":SECONDARY}), md=2),
            dbc.Col(html.Div([
                html.P("After", style={"fontWeight":"700","color":SECONDARY,"fontSize":"12px","textTransform":"uppercase"}),
                html.H5(f"{new_r:,} rows × {new_c} cols", style={"fontWeight":"800","color":SUCCESS}),
            ], style={**CARD,"textAlign":"center","padding":"14px"}), md=5),
        ], className="mb-3", align="center"),
        dbc.Row([
            dbc.Col(html.Div([
                html.P("Columns dropped", style={"fontWeight":"700","fontSize":"12px"}),
                html.P(", ".join(dropped) if dropped else "None",
                       style={"fontSize":"12px","color":SECONDARY}),
            ], style=CARD), md=6),
            dbc.Col(html.Div([
                html.P("Duplicate rows removed", style={"fontWeight":"700","fontSize":"12px"}),
                html.P(f"{stats['rows_dropped']:,}",
                       style={"fontSize":"22px","fontWeight":"800","color":PRIMARY}),
            ], style=CARD), md=6),
        ]),
        html.Div([
            html.H6("Cleaned Data Preview", style={"fontWeight":"700"}),
            dash_table.DataTable(
                data=df_c.head(8).astype(str).to_dict("records"),
                columns=[{"name":c,"id":c} for c in df_c.columns],
                style_table={"overflowX":"auto"},
                style_header={"backgroundColor":SUCCESS,"color":"white","fontSize":"11px","fontWeight":"700"},
                style_cell={"fontSize":"11px","padding":"6px 10px","maxWidth":"160px",
                            "overflow":"hidden","textOverflow":"ellipsis"},
            ),
        ], style=CARD),
    ])
    return df_to_store(df_c), results, True, False


# ── Populate target dropdown once cleaned data is ready ─────
@app.callback(
    Output("train-target","options"),
    Output("train-target","value"),
    Input("store-cleaned","data"),
    prevent_initial_call=True,
)
def populate_target(cleaned):
    if not cleaned: return [], None
    df_c = store_to_df(cleaned)
    num_options = df_c.select_dtypes(include=np.number).columns.tolist()
    default = "matched_score" if "matched_score" in num_options else (num_options[0] if num_options else None)
    opts = [{"label":c,"value":c} for c in num_options]
    return opts, default


# ── Training ─────────────────────────────────────────────────
@app.callback(
    Output("train-results","children"),
    Output("store-model-meta","data"),
    Input("btn-run-train","n_clicks"),
    State("store-cleaned","data"),
    State("train-target","value"),
    State("train-models","value"),
    State("train-test-size","value"),
    prevent_initial_call=True,
)
def run_training_cb(n, cleaned, target, models_sel, test_size):
    if not cleaned:
        return dbc.Alert("Run Preprocessing first.", color="warning"), no_update
    if not target:
        return dbc.Alert("Select a target column.", color="warning"), no_update
    if not models_sel:
        return dbc.Alert("Select at least one model.", color="warning"), no_update

    df_c = store_to_df(cleaned)
    if target not in df_c.columns:
        return dbc.Alert(f"Column '{target}' not found in cleaned data.", color="danger"), no_update

    try:
        res = train_models(df_c, target, models_sel, test_size)
    except Exception as e:
        return dbc.Alert(f"Training error: {e}", color="danger"), no_update

    # save model server-side for Predict mode
    _model_store["model"]    = res["model_obj"]
    _model_store["features"] = res["feats"]
    _model_store["target"]   = target

    # results table
    perf = pd.DataFrame(res["results"]).T.reset_index().rename(columns={"index":"Model"})
    best = res["best"]

    # actual vs predicted scatter
    fig_scatter = go.Figure()
    if res["preds"] and res["y_test"]:
        fig_scatter.add_trace(go.Scatter(
            x=res["y_test"], y=res["preds"],
            mode="markers", marker=dict(color=PRIMARY, opacity=0.4, size=5),
            name=best,
        ))
        lim = [min(res["y_test"]), max(res["y_test"])]
        fig_scatter.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                          line=dict(color=DANGER, dash="dash", width=1.5),
                                          name="Perfect"))
    fig_scatter.update_layout(
        xaxis_title="Actual", yaxis_title="Predicted",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=20,r=20,t=10,b=30), height=300,
    )

    # feature importance
    imp_fig = go.Figure()
    if res["importances"]:
        imp = pd.Series(res["importances"]).sort_values(ascending=True).tail(12)
        imp_fig.add_trace(go.Bar(x=imp.values, y=imp.index, orientation="h",
                                  marker_color=PRIMARY))
        imp_fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=20,r=20,t=10,b=20), height=300,
                               xaxis_title="Importance")

    return html.Div([
        dbc.Alert(f"✅  Training complete!  Best model: {best}", color="success",
                  style={"padding":"10px 16px","marginBottom":"12px"}),
        # model comparison
        html.Div([
            html.H6("Model Performance", style={"fontWeight":"700"}),
            dash_table.DataTable(
                data=perf.to_dict("records"),
                columns=[{"name":c,"id":c} for c in perf.columns],
                style_header={"backgroundColor":SUCCESS,"color":"white","fontWeight":"700","fontSize":"13px"},
                style_cell={"fontSize":"13px","padding":"8px 16px","textAlign":"center"},
                style_data_conditional=[
                    {"if":{"filter_query":f"{{Model}} = '{best}'"},
                     "backgroundColor":"#e8fff5","fontWeight":"700"},
                    {"if":{"row_index":"odd"},"backgroundColor":"#f8f9fa"},
                ],
            ),
        ], style=CARD),
        # charts
        dbc.Row([
            dbc.Col(html.Div([
                html.H6(f"Actual vs Predicted  ({best})", style={"fontWeight":"700"}),
                dcc.Graph(figure=fig_scatter, style={"height":"300px"}),
            ], style=CARD), md=6),
            dbc.Col(html.Div([
                html.H6("Feature Importance", style={"fontWeight":"700"}),
                dcc.Graph(figure=imp_fig, style={"height":"300px"}),
            ], style=CARD) if res["importances"] else html.Div(), md=6),
        ]),
        # feature list
        html.Div([
            html.P(f"Features used ({len(res['feats'])}):  " +
                   ", ".join(res["feats"][:30]) + ("…" if len(res["feats"])>30 else ""),
                   style={"fontSize":"12px","color":SECONDARY}),
        ], style=CARD),
    ]), {"feats": res["feats"], "target": target, "best": best}


# ── Toggle Train / Predict sections ──────────────────────────
@app.callback(
    Output("section-train",   "style"),
    Output("section-predict", "style"),
    Input("step4-mode", "value"),
)
def toggle_step4_mode(mode):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (show, hide) if mode == "train" else (hide, show)


# ── Populate predict "compare against" dropdown ──────────────
@app.callback(
    Output("predict-actual-col", "options"),
    Output("predict-model-status", "children"),
    Input("store-step", "data"),
    State("store-cleaned", "data"),
    State("store-model-meta", "data"),
    prevent_initial_call=True,
)
def update_predict_panel(step, cleaned, meta):
    if step != 4 or not cleaned:
        return [], no_update
    df_c = store_to_df(cleaned)
    num_opts = [{"label": c, "value": c}
                for c in df_c.select_dtypes(include=np.number).columns]

    if _model_store["model"] is not None:
        status = dbc.Alert(
            [html.Strong(f"✅  Model ready: "), f"{meta['best']}  ·  "
             f"{len(meta['feats'])} features  ·  target: {meta['target']}"],
            color="success", style={"padding": "10px 16px"},
        )
    else:
        status = dbc.Alert(
            "ℹ️  No model trained yet — clicking Run will auto-train XGBoost on this dataset.",
            color="info", style={"padding": "10px 16px"},
        )
    return num_opts, status


# ── Predict callback ─────────────────────────────────────────
@app.callback(
    Output("predict-results",   "children"),
    Output("btn-download-pred", "disabled"),
    Input("btn-run-predict", "n_clicks"),
    State("store-cleaned", "data"),
    State("predict-actual-col", "value"),
    prevent_initial_call=True,
)
def run_predict_cb(n, cleaned, actual_col):
    if not cleaned:
        return dbc.Alert("Run Preprocessing first.", color="warning"), True

    df_c = store_to_df(cleaned)

    # use existing model or auto-train XGBoost
    if _model_store["model"] is not None:
        model   = _model_store["model"]
        feats   = _model_store["features"]
        model_label = f"{_model_store['target']} model (already trained)"
    else:
        from xgboost import XGBRegressor
        num_cols_c = df_c.select_dtypes(include=np.number).columns.tolist()
        if not num_cols_c:
            return dbc.Alert("No numeric columns found to predict on.", color="danger"), True
        # pick last numeric column as target if no training done
        target_auto = num_cols_c[-1]
        feats = [c for c in num_cols_c if c != target_auto and df_c[c].std() > 0]
        if not feats:
            return dbc.Alert("Not enough numeric features to predict.", color="danger"), True
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                             random_state=42, n_jobs=-1, verbosity=0)
        model.fit(df_c[feats].fillna(0), df_c[target_auto])
        model_label = f"auto-trained XGBoost on '{target_auto}'"
        _model_store["model"]    = model
        _model_store["features"] = feats
        _model_store["target"]   = target_auto

    # align features — drop any missing from this dataset
    avail_feats = [f for f in feats if f in df_c.columns]
    if not avail_feats:
        return dbc.Alert(
            "The trained model's features are not present in this dataset.", color="danger"), True

    preds = model.predict(df_c[avail_feats].fillna(0))
    out = df_c.copy()
    out["predicted_score"] = preds.round(4)

    # save to store for download
    _model_store["last_predictions"] = out

    # build scatter if actual column chosen
    scatter_card = html.Div()
    if actual_col and actual_col in out.columns:
        r2  = float(np.corrcoef(out[actual_col], out["predicted_score"])[0,1]**2)
        mae = float(np.abs(out[actual_col] - out["predicted_score"]).mean())
        fig_s = px.scatter(
            out.sample(min(3000, len(out)), random_state=42),
            x=actual_col, y="predicted_score",
            opacity=0.4, color_discrete_sequence=[PRIMARY],
            labels={actual_col: f"Actual ({actual_col})", "predicted_score": "Predicted"},
        )
        lim = [float(out[actual_col].min()), float(out[actual_col].max())]
        fig_s.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                    line=dict(color=DANGER, dash="dash", width=1.5),
                                    name="Perfect"))
        fig_s.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                             margin=dict(l=20,r=20,t=10,b=20), height=300)
        scatter_card = html.Div([
            html.H6(f"Predicted vs Actual  ·  R²={r2:.3f}  MAE={mae:.4f}",
                    style={"fontWeight":"700"}),
            dcc.Graph(figure=fig_s, style={"height":"300px"}),
        ], style=CARD)

    # preview table (top 20)
    preview_cols = [c for c in ([actual_col] if actual_col else []) +
                    ["predicted_score"] + avail_feats[:5] if c in out.columns]
    preview = out[preview_cols].head(20).round(4)

    return html.Div([
        dbc.Alert(
            [html.Strong("✅  Predictions ready  ·  "),
             f"{len(out):,} rows  ·  using {model_label}"],
            color="success", style={"padding":"10px 16px","marginBottom":"12px"},
        ),
        scatter_card,
        html.Div([
            html.H6("Preview (top 20 rows)", style={"fontWeight":"700"}),
            dash_table.DataTable(
                data=preview.astype(str).to_dict("records"),
                columns=[{"name":c,"id":c} for c in preview.columns],
                style_table={"overflowX":"auto"},
                style_header={"backgroundColor":PRIMARY,"color":"white",
                              "fontSize":"11px","fontWeight":"700"},
                style_cell={"fontSize":"11px","padding":"6px 10px",
                            "maxWidth":"160px","overflow":"hidden","textOverflow":"ellipsis"},
                style_data_conditional=[
                    {"if":{"column_id":"predicted_score"},
                     "backgroundColor":"#e8f4ff","fontWeight":"700"},
                    {"if":{"row_index":"odd"},"backgroundColor":"#f8f9fa"},
                ],
            ),
        ], style=CARD),
        html.P("Click ⬇️ Download CSV above to get the full predictions file.",
               style={"fontSize":"12px","color":SECONDARY}),
    ]), False


# ── Download predictions CSV ──────────────────────────────────
@app.callback(
    Output("download-predictions", "data"),
    Input("btn-download-pred", "n_clicks"),
    prevent_initial_call=True,
)
def download_predictions(n):
    df_out = _model_store.get("last_predictions")
    if df_out is None:
        return no_update
    return dcc.send_data_frame(df_out.to_csv, "predictions.csv", index=False)


# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
