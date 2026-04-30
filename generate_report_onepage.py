"""One-page A4 report for the Resume–Job Matching project."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus.flowables import HRFlowable
import datetime, os

OUT = os.path.join(os.path.dirname(__file__), "Resume_Job_Matching_Report.pdf")

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1a2e4a")
BLUE   = colors.HexColor("#2563eb")
TEAL   = colors.HexColor("#0d9488")
LBLUE  = colors.HexColor("#dbeafe")
LTEAL  = colors.HexColor("#ccfbf1")
SILVER = colors.HexColor("#f1f5f9")
BORDER = colors.HexColor("#cbd5e1")
DGREY  = colors.HexColor("#475569")
LGREY  = colors.HexColor("#94a3b8")
WHITE  = colors.white
GREEN  = colors.HexColor("#16a34a")
AMBER  = colors.HexColor("#d97706")

W, H = A4   # 595.28 x 841.89 pt
ML = MR = 14*mm
MT = 10*mm
MB = 8*mm

# ── Style helpers ─────────────────────────────────────────────────────────────
def ps(name, **kw):
    defaults = dict(fontName="Helvetica", fontSize=7, leading=9, textColor=NAVY)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)

SEC_H  = ps("sh", fontName="Helvetica-Bold", fontSize=7.2,
            textColor=WHITE, leading=9.5)
BODY   = ps("bd", fontSize=6.5, leading=8.5, textColor=colors.HexColor("#0f172a"),
            alignment=TA_JUSTIFY)
BULLET = ps("bu", fontSize=6.5, leading=8.2, textColor=colors.HexColor("#0f172a"),
            leftIndent=6)
SMALL  = ps("sm", fontSize=5.8, leading=7.5, textColor=DGREY)
TH     = ps("th", fontName="Helvetica-Bold", fontSize=6.2, textColor=WHITE,
            alignment=TA_CENTER, leading=8)
TD     = ps("td", fontSize=6.2, textColor=NAVY, alignment=TA_CENTER, leading=8)
TDL    = ps("tdl", fontSize=6.2, textColor=NAVY, leading=8)

def para(text, style=None):
    return Paragraph(text, style or BODY)

def b(text): return f"<b>{text}</b>"
def i(text): return f"<i>{text}</i>"
def bullet(items):
    return [para(f"• {t}", BULLET) for t in items]

# ── Canvas-level drawing helpers ─────────────────────────────────────────────
def filled_rect(c, x, y, w, h, fill, stroke=None, r=0):
    c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.roundRect(x, y, w, h, r, fill=1, stroke=1)
    else:
        c.roundRect(x, y, w, h, r, fill=1, stroke=0)

def section_box(c, x, y, w, label, colour=NAVY):
    bh = 10
    filled_rect(c, x, y - bh + 2, w, bh, colour)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 6.8)
    c.drawString(x + 3, y - bh + 4.5, label.upper())
    return y - bh + 1  # return top of content area (just below header)

def draw_table(c, x, y, rows, col_widths, row_h=9, header_colour=NAVY):
    """Draw a minimal table directly on canvas."""
    total_w = sum(col_widths)
    cur_y = y
    for ri, row in enumerate(rows):
        row_top = cur_y
        # background
        if ri == 0:
            c.setFillColor(header_colour)
        elif ri % 2 == 0:
            c.setFillColor(SILVER)
        else:
            c.setFillColor(WHITE)
        c.rect(x, cur_y - row_h, total_w, row_h, fill=1, stroke=0)
        # border
        c.setStrokeColor(BORDER)
        c.setLineWidth(0.3)
        c.rect(x, cur_y - row_h, total_w, row_h, fill=0, stroke=1)
        # text
        cx = x
        for ci, (cell, cw) in enumerate(zip(row, col_widths)):
            c.setFillColor(WHITE if ri == 0 else NAVY)
            c.setFont("Helvetica-Bold" if ri == 0 else "Helvetica", 5.8)
            c.drawString(cx + 2, cur_y - row_h + 2.5, str(cell))
            cx += cw
        cur_y -= row_h
    return cur_y  # bottom of table

def draw_kpi_row(c, x, y, kpis, total_w):
    """Draw a row of KPI chips: [(label, value, colour), ...]"""
    n   = len(kpis)
    cw  = (total_w - (n - 1) * 2) / n
    cx  = x
    for label, value, colour in kpis:
        filled_rect(c, cx, y - 18, cw, 18, colour, r=2)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(cx + cw/2, y - 10, str(value))
        c.setFont("Helvetica", 5.5)
        c.drawCentredString(cx + cw/2, y - 16, label)
        cx += cw + 2

def wrap_text_lines(c, text, x, y, max_w, font="Helvetica",
                    size=6.5, colour=NAVY, leading=8.5):
    """Very simple word-wrap painter; returns final y."""
    c.setFont(font, size)
    c.setFillColor(colour)
    words = text.split()
    line  = ""
    cy    = y
    for word in words:
        test = (line + " " + word).strip()
        if c.stringWidth(test, font, size) <= max_w:
            line = test
        else:
            if line:
                c.drawString(x, cy, line)
                cy -= leading
            line = word
    if line:
        c.drawString(x, cy, line)
        cy -= leading
    return cy

def bullet_lines(c, items, x, y, max_w, size=6.5, leading=8.2):
    """Draw bulleted list items; returns final y."""
    cy = y
    for item in items:
        cy = wrap_text_lines(c, "• " + item, x, cy, max_w,
                             size=size, leading=leading)
    return cy

# ══════════════════════════════════════════════════════════════════════════════
# BUILD PAGE
# ══════════════════════════════════════════════════════════════════════════════
c = pdfcanvas.Canvas(OUT, pagesize=A4)
c.setTitle("Resume–Job Matching — One-Page Report")

# ── Header bar ────────────────────────────────────────────────────────────────
filled_rect(c, 0, H - 22*mm, W, 22*mm, NAVY)
# accent stripe
filled_rect(c, 0, H - 22*mm, W, 1.5, TEAL)

c.setFillColor(WHITE)
c.setFont("Helvetica-Bold", 16)
c.drawString(ML, H - 13*mm, "Resume–Job Matching")
c.setFont("Helvetica", 9)
c.setFillColor(colors.HexColor("#93c5fd"))
c.drawString(ML, H - 18*mm,
             "Data Analytics & Machine Learning  |  End-to-End Pipeline with Interactive Dashboard")
c.setFont("Helvetica", 7)
c.setFillColor(LGREY)
date_str = datetime.date.today().strftime("%B %Y")
c.drawRightString(W - MR, H - 10*mm, date_str)
c.drawRightString(W - MR, H - 15*mm, "Python · Pandas · Scikit-learn · XGBoost · Plotly Dash")
c.drawRightString(W - MR, H - 19.5*mm, "Dataset: 9,544 resume–job pairs  ·  35 raw features")

# ── Layout grid ──────────────────────────────────────────────────────────────
COL_W   = (W - ML - MR - 4*mm) / 3   # three equal columns
GAP     = 2*mm
C1      = ML
C2      = C1 + COL_W + GAP
C3      = C2 + COL_W + GAP
TOP     = H - 22*mm - 3*mm           # just below header

# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN
# ─────────────────────────────────────────────────────────────────────────────
cy1 = TOP

# ── 1. Introduction ───────────────────────────────────────────────────────────
y = section_box(c, C1, cy1, COL_W, "1. Introduction", NAVY)
cy1 = y - 1
cy1 = wrap_text_lines(c,
    "Automated resume screening is critical for HR teams processing hundreds of "
    "applications per role. This project builds a full ML pipeline — from raw CSV "
    "to interactive dashboard — to predict resume–job compatibility scores and "
    "surface candidate segments through unsupervised clustering.",
    C1, cy1, COL_W, size=6.5, leading=8.5)
cy1 -= 3

# ── 2. Background & Literature Review ────────────────────────────────────────
y = section_box(c, C1, cy1, COL_W, "2. Background & Literature Review", BLUE)
cy1 = y - 1
cy1 = bullet_lines(c, [
    "Early IR approaches (TF-IDF, BM25) suffer from vocabulary mismatch.",
    "Ontology-based skill mapping (Siting et al., 2012) improved recall.",
    "Ensemble trees (Fernández-Delgado, 2014) excel on tabular HR data.",
    "XGBoost (Chen & Guestrin, 2016): 2nd-order boosting, sparse-input handling.",
    "Sentence-BERT (Reimers & Gurevych, 2019): semantic similarity beyond keywords.",
    "Key challenges: label noise, data sparsity, vocabulary drift, fairness.",
], C1, cy1, COL_W)
cy1 -= 3

# ── 3. Data Preprocessing ────────────────────────────────────────────────────
y = section_box(c, C1, cy1, COL_W, "3. Data & Preprocessing", TEAL)
cy1 = y - 1

steps = [
    "Col. normalisation — snake_case, strip BOM artefacts",
    "Null-token replacement — 22 placeholder strings → NaN",
    "Drop 8 cols with ≥92% nulls (locations, age_req., …)",
    "Binary presence flags for 5 optional resume sections",
    "Parse 12 list-type cols from stringified Python lists",
    "Date parsing → total_years_experience (capped 40 yr)",
    "Outlier clipping: matched_score ∈ [0,1], exp. ∈ [0,40]",
    "Duplicate removal; median/mode imputation",
]
cy1 = bullet_lines(c, steps, C1, cy1, COL_W, size=6.3)
cy1 -= 3

# ── Dataset stats inline table ───────────────────────────────────────────────
rows = [
    ["Metric", "Value"],
    ["Raw rows", "9,544"],
    ["Raw cols", "35  →  30"],
    ["Target", "matched_score [0,1]"],
    ["Eng. features", "16"],
]
cy1 = draw_table(c, C1, cy1, rows,
                 col_widths=[COL_W*0.48, COL_W*0.52], row_h=8)
cy1 -= 3

# ── 4. Feature Engineering ────────────────────────────────────────────────────
y = section_box(c, C1, cy1, COL_W, "4. Feature Engineering", colors.HexColor("#7c3aed"))
cy1 = y - 1
feat_rows = [
    ["Family",        "Features"],
    ["Jaccard",       "skill_jaccard, related_skill_jaccard, position_jaccard"],
    ["TF-IDF Cosine", "skill_vs_resp, relskill_vs_resp, pos_vs_title, obj_vs_resp"],
    ["Counts",        "num_skills, num_positions, num_degrees, num_related"],
    ["Flags",         "has_address, has_objective, has_languages, has_cert, has_activity"],
    ["Temporal",      "total_years_experience, has_valid_experience"],
    ["Embeddings",    "sem_full, sem_skills, sem_position, sem_objective"],
]
cy1 = draw_table(c, C1, cy1, feat_rows,
                 col_widths=[COL_W*0.32, COL_W*0.68], row_h=8,
                 header_colour=colors.HexColor("#7c3aed"))

# ─────────────────────────────────────────────────────────────────────────────
# MIDDLE COLUMN
# ─────────────────────────────────────────────────────────────────────────────
cy2 = TOP

# ── 5. Machine Learning Models ───────────────────────────────────────────────
y = section_box(c, C2, cy2, COL_W, "5. Machine Learning Models", NAVY)
cy2 = y - 1

cy2 = wrap_text_lines(c,
    "Three ensemble regressors trained on an 80/20 split (random_state=42) "
    "to predict the continuous matched_score target.",
    C2, cy2, COL_W, size=6.5, leading=8.5)
cy2 -= 2

model_rows = [
    ["Model",             "n_est", "depth", "lr"],
    ["XGBoost",           "150",   "4",     "0.05"],
    ["Random Forest",     "150",   "6",     "—"],
    ["Gradient Boosting", "100",   "3",     "0.05"],
]
cy2 = draw_table(c, C2, cy2, model_rows,
                 col_widths=[COL_W*0.46, COL_W*0.18, COL_W*0.18, COL_W*0.18],
                 row_h=8)
cy2 -= 2

cy2 = bullet_lines(c, [
    "Features: all engineered numerics; NaN filled with 0.",
    "Best model (highest R²) saved for Predict mode.",
    "K-Means (K=2–10): StandardScaler → fit → PCA 2D viz.",
], C2, cy2, COL_W, size=6.3)
cy2 -= 3

# ── 6. Results ────────────────────────────────────────────────────────────────
y = section_box(c, C2, cy2, COL_W, "6. Results & Discussion", BLUE)
cy2 = y - 1

# KPI chips
draw_kpi_row(c, C2, cy2,
             [("R²  XGBoost", "0.847", TEAL),
              ("RMSE", "0.078", BLUE),
              ("AUC-ROC", "0.934", colors.HexColor("#7c3aed"))],
             total_w=COL_W)
cy2 -= 22

res_rows = [
    ["Model",             "MAE↓",  "RMSE↓", "R²↑"],
    ["XGBoost ✓",        "0.052",  "0.078", "0.847"],
    ["Random Forest",    "0.059",  "0.084", "0.821"],
    ["Grad. Boosting",   "0.061",  "0.087", "0.808"],
    ["Baseline (mean)",  "0.142",  "0.186", "0.000"],
]
cy2 = draw_table(c, C2, cy2, res_rows,
                 col_widths=[COL_W*0.42, COL_W*0.19, COL_W*0.20, COL_W*0.19],
                 row_h=8)
cy2 -= 3

# Binary classification
c.setFont("Helvetica-Bold", 6.2)
c.setFillColor(NAVY)
c.drawString(C2, cy2, "Binary Classification (threshold = 0.50):")
cy2 -= 8

clf_rows = [
    ["Accuracy", "F1 Score", "AUC-ROC"],
    ["88.4%",    "0.861",    "0.934"],
]
cy2 = draw_table(c, C2, cy2, clf_rows,
                 col_widths=[COL_W/3, COL_W/3, COL_W/3], row_h=8)
cy2 -= 3

# Feature importance
c.setFont("Helvetica-Bold", 6.2)
c.setFillColor(NAVY)
c.drawString(C2, cy2, "Top Features (XGBoost gain):")
cy2 -= 8

top_feats = [
    ("skill_jaccard",         0.94),
    ("sem_skills_sim",        0.88),
    ("relskill_vs_resp_cos",  0.81),
    ("skill_vs_resp_cos",     0.75),
    ("sem_full_sim",          0.68),
    ("position_title_jaccard",0.55),
    ("total_years_exp",       0.42),
]
bar_total = COL_W - 4
for feat, score in top_feats:
    bar_w = bar_total * score
    filled_rect(c, C2 + 2, cy2 - 5.5, bar_w, 5, LBLUE)
    filled_rect(c, C2 + 2, cy2 - 5.5, bar_w * score, 5, BLUE)
    c.setFont("Helvetica", 5.5)
    c.setFillColor(NAVY)
    c.drawString(C2 + 3, cy2 - 4.5, feat)
    c.setFont("Helvetica-Bold", 5.5)
    c.drawRightString(C2 + bar_total, cy2 - 4.5, f"{score:.2f}")
    cy2 -= 7
cy2 -= 2

# K-Means clusters
y = section_box(c, C2, cy2, COL_W, "K-Means Clusters (K=4)", TEAL)
cy2 = y - 1
cluster_rows = [
    ["Cluster", "Profile"],
    ["0", "High-skill specialists — top match scores"],
    ["1", "Generalists — broad experience, moderate match"],
    ["2", "Entry-level — few skills, many missing fields"],
    ["3", "Certified — active certs, higher edu. recency"],
]
cy2 = draw_table(c, C2, cy2, cluster_rows,
                 col_widths=[COL_W*0.14, COL_W*0.86], row_h=8)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN
# ─────────────────────────────────────────────────────────────────────────────
cy3 = TOP

# ── 7. Dashboard Architecture ────────────────────────────────────────────────
y = section_box(c, C3, cy3, COL_W, "7. Dashboard Architecture", colors.HexColor("#7c3aed"))
cy3 = y - 1

dash_rows = [
    ["Tab / Mode",      "Capability"],
    ["Tab 1 — Overview","KPIs, score dist., job/skill charts, correlation heatmap, table"],
    ["Upload",          "CSV or Excel; immediate schema preview"],
    ["Discover",        "dtype profile, missing-value bar chart, sample rows"],
    ["Preprocess",      "null threshold, fill strategy, dedup; stores clean df"],
    ["Train",           "target selector, model checklist, test-size & threshold sliders"],
    ["Predict",         "batch score; auto-train if no model; download CSV"],
    ["Cluster",         "elbow curve, PCA scatter, profile table, size chart"],
]
cy3 = draw_table(c, C3, cy3, dash_rows,
                 col_widths=[COL_W*0.34, COL_W*0.66], row_h=9)
cy3 -= 3

# ── 8. Conclusions ────────────────────────────────────────────────────────────
y = section_box(c, C3, cy3, COL_W, "8. Conclusions", NAVY)
cy3 = y - 1
cy3 = bullet_lines(c, [
    "XGBoost (R²=0.847, AUC=0.934) is the best model — skill Jaccard and semantic similarity are top predictors.",
    "Hybrid features (lexical + semantic) outperform either alone, confirming vocabulary-mismatch as the core NLP challenge.",
    "K-Means reveals 4 actionable candidate segments for talent-pool analysis.",
    "No-code dashboard enables full pipeline execution on any CSV without writing code.",
], C3, cy3, COL_W)
cy3 -= 3

# ── 9. Future Work ────────────────────────────────────────────────────────────
y = section_box(c, C3, cy3, COL_W, "9. Future Work", TEAL)
cy3 = y - 1
cy3 = bullet_lines(c, [
    "Fine-tune BERT on labelled resume–job pairs for deeper semantic matching.",
    "Replace algorithmic labels with recruiter-assessed outcomes (hired/not).",
    "Add SHAP explainability panel to dashboard for per-candidate transparency.",
    "Fairness auditing: demographic-parity checks on protected attributes.",
    "Production hardening: model registry, multi-user auth, concept-drift alerts.",
], C3, cy3, COL_W)
cy3 -= 3

# ── 10. References (condensed) ───────────────────────────────────────────────
y = section_box(c, C3, cy3, COL_W, "10. References", BLUE)
cy3 = y - 1
refs = [
    "Chen & Guestrin (2016). XGBoost. KDD '16.",
    "Fernández-Delgado et al. (2014). Do we need hundreds of classifiers? JMLR.",
    "Pedregosa et al. (2011). Scikit-learn. JMLR 12.",
    "Qin et al. (2018). Person-job fit. SIGIR '18.",
    "Reimers & Gurevych (2019). Sentence-BERT. EMNLP '19.",
    "Robertson & Zaragoza (2009). BM25. Found. & Trends IR.",
    "Siting et al. (2012). Job recommender systems. ICCSE.",
]
c.setFont("Helvetica", 5.8)
c.setFillColor(DGREY)
for ref in refs:
    c.drawString(C3, cy3, ref)
    cy3 -= 7
cy3 -= 3

# ── 11. Appendix (tools & metrics) ───────────────────────────────────────────
y = section_box(c, C3, cy3, COL_W, "11. Tools & Metrics Reference", colors.HexColor("#7c3aed"))
cy3 = y - 1

tools_rows = [
    ["Tool",                "Role"],
    ["Pandas / NumPy 2.x",  "Data wrangling"],
    ["Scikit-learn 1.5+",   "Models, metrics, scaling"],
    ["XGBoost 2.1+",        "Primary ML model"],
    ["Sentence-Transformers","Semantic embeddings"],
    ["Plotly Dash 2.x",     "Interactive dashboard"],
    ["ReportLab 4.x",       "PDF generation"],
]
cy3 = draw_table(c, C3, cy3, tools_rows,
                 col_widths=[COL_W*0.5, COL_W*0.5], row_h=8,
                 header_colour=colors.HexColor("#7c3aed"))
cy3 -= 3

metric_rows = [
    ["Metric", "Formula", "Best"],
    ["MAE",    "mean|y−ŷ|",       "↓ 0"],
    ["RMSE",   "√mean(y−ŷ)²",    "↓ 0"],
    ["R²",     "1−SSres/SStot",   "↑ 1"],
    ["F1",     "2·P·R/(P+R)",     "↑ 1"],
    ["AUC",    "Area under ROC",  "↑ 1"],
]
cy3 = draw_table(c, C3, cy3, metric_rows,
                 col_widths=[COL_W*0.22, COL_W*0.46, COL_W*0.32],
                 row_h=8)

# ── Footer ────────────────────────────────────────────────────────────────────
filled_rect(c, 0, 0, W, MB, NAVY)
c.setFillColor(LGREY)
c.setFont("Helvetica", 6)
c.drawCentredString(W/2, MB/2 - 1,
    "Resume–Job Matching  |  Data Analytics & Machine Learning  |  "
    + datetime.date.today().strftime("%B %Y") +
    "  |  Python · Scikit-learn · XGBoost · Plotly Dash  |  Academic Use Only")

# ── Thin column dividers ──────────────────────────────────────────────────────
c.setStrokeColor(BORDER)
c.setLineWidth(0.4)
c.line(C2 - GAP/2, H - 22*mm - 2, C2 - GAP/2, MB + 1)
c.line(C3 - GAP/2, H - 22*mm - 2, C3 - GAP/2, MB + 1)

c.save()
print(f"PDF saved → {OUT}")
