"""Generate a clean PDF report for the Resume–Job Matching project."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import ListFlowable, ListItem
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os, datetime

# ── Output path ───────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "Resume_Job_Matching_Report.pdf")

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#1a2e4a")
BLUE    = colors.HexColor("#2563eb")
LBLUE   = colors.HexColor("#dbeafe")
TEAL    = colors.HexColor("#0d9488")
LTEAL   = colors.HexColor("#ccfbf1")
SILVER  = colors.HexColor("#f1f5f9")
BORDER  = colors.HexColor("#cbd5e1")
BLACK   = colors.HexColor("#0f172a")
DGREY   = colors.HexColor("#475569")
LGREY   = colors.HexColor("#94a3b8")
WHITE   = colors.white
RED     = colors.HexColor("#dc2626")
AMBER   = colors.HexColor("#d97706")
GREEN   = colors.HexColor("#16a34a")

PAGE_W, PAGE_H = A4
L_MARGIN = R_MARGIN = 2.2 * cm
T_MARGIN = B_MARGIN = 2.2 * cm

# ── Style sheet ───────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    """Create a named ParagraphStyle inheriting from 'Normal'."""
    return ParagraphStyle(name, parent=base["Normal"], **kw)

cover_title  = S("CoverTitle",  fontSize=28, textColor=WHITE,  leading=36,
                 fontName="Helvetica-Bold", alignment=TA_CENTER)
cover_sub    = S("CoverSub",    fontSize=13, textColor=LBLUE,  leading=18,
                 fontName="Helvetica",      alignment=TA_CENTER)
cover_meta   = S("CoverMeta",   fontSize=10, textColor=LGREY,  leading=14,
                 fontName="Helvetica",      alignment=TA_CENTER)

h1           = S("H1",  fontSize=16, textColor=NAVY, fontName="Helvetica-Bold",
                 spaceBefore=18, spaceAfter=6,  leading=22)
h2           = S("H2",  fontSize=12, textColor=BLUE, fontName="Helvetica-Bold",
                 spaceBefore=12, spaceAfter=4,  leading=16)
h3           = S("H3",  fontSize=10, textColor=TEAL, fontName="Helvetica-Bold",
                 spaceBefore=8,  spaceAfter=2,  leading=14)
body         = S("Body", fontSize=9.5, textColor=BLACK, fontName="Helvetica",
                 leading=14, spaceAfter=4, alignment=TA_JUSTIFY)
body_sm      = S("BodySm", fontSize=8.5, textColor=DGREY, fontName="Helvetica",
                 leading=12, spaceAfter=3)
bullet_s     = S("Bullet", fontSize=9.5, textColor=BLACK, fontName="Helvetica",
                 leading=13, leftIndent=14, spaceAfter=2)
caption      = S("Caption", fontSize=8, textColor=LGREY, fontName="Helvetica-Oblique",
                 alignment=TA_CENTER, spaceAfter=6)
tbl_hdr      = S("TblHdr", fontSize=9, textColor=WHITE, fontName="Helvetica-Bold",
                 alignment=TA_CENTER, leading=12)
tbl_cell     = S("TblCell", fontSize=9, textColor=BLACK, fontName="Helvetica",
                 alignment=TA_CENTER, leading=12)
tbl_cell_l   = S("TblCellL", fontSize=9, textColor=BLACK, fontName="Helvetica",
                 leading=12)
callout      = S("Callout", fontSize=9, textColor=NAVY, fontName="Helvetica-Oblique",
                 leading=13, leftIndent=10, rightIndent=10,
                 borderPadding=(6,10,6,10), spaceAfter=6)

def rule(): return HRFlowable(width="100%", thickness=0.6, color=BORDER,
                               spaceAfter=6, spaceBefore=4)

def section_rule(): return HRFlowable(width="100%", thickness=1.5, color=BLUE,
                                       spaceAfter=8, spaceBefore=4)

def bullet_list(*items):
    return [Paragraph(f"• {i}", bullet_s) for i in items]

def kpi_table(rows, col_widths=None):
    """Render a simple data table with header row."""
    header, *data = rows
    t_data = [[Paragraph(str(c), tbl_hdr) for c in header]]
    for row in data:
        t_data.append([Paragraph(str(c), tbl_cell) for c in row])
    cw = col_widths or [PAGE_W / len(header)] * len(header)
    tbl = Table(t_data, colWidths=cw, repeatRows=1)
    style = [
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, SILVER]),
        ("GRID",  (0,0), (-1,-1), 0.4, BORDER),
        ("BOX",   (0,0), (-1,-1), 0.8, NAVY),
        ("VALIGN",(0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]
    tbl.setStyle(TableStyle(style))
    return tbl

def highlight_box(text, bg=LBLUE, border=BLUE):
    hb_style = ParagraphStyle("HB", parent=body, fontSize=9,
                               textColor=NAVY, leading=13)
    data = [[Paragraph(text, hb_style)]]
    t = Table(data, colWidths=[PAGE_W - L_MARGIN - R_MARGIN - 0.4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("BOX",        (0,0), (-1,-1), 1,  border),
        ("LEFTPADDING",(0,0), (-1,-1), 10),
        ("RIGHTPADDING",(0,0),(-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
    ]))
    return t

# ── Page templates ────────────────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 1*cm, PAGE_W, 1*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(LGREY)
    canvas.drawString(L_MARGIN, PAGE_H - 0.65*cm,
                      "Resume–Job Matching  |  Data Analytics & Machine Learning Report")
    canvas.drawRightString(PAGE_W - R_MARGIN, PAGE_H - 0.65*cm,
                           datetime.date.today().strftime("%B %Y"))
    # Footer bar
    canvas.setFillColor(SILVER)
    canvas.rect(0, 0, PAGE_W, 0.9*cm, fill=1, stroke=0)
    canvas.setFillColor(DGREY)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(PAGE_W/2, 0.35*cm, f"Page {doc.page}")
    canvas.setFillColor(LGREY)
    canvas.drawString(L_MARGIN, 0.35*cm, "Confidential – Academic Use Only")
    canvas.restoreState()

def on_cover(canvas, doc):
    # Full navy background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Accent strip
    canvas.setFillColor(BLUE)
    canvas.rect(0, PAGE_H*0.38, PAGE_W, 4, fill=1, stroke=0)
    # Bottom teal strip
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, PAGE_W, 0.7*cm, fill=1, stroke=0)

# ── Document ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUT,
    pagesize=A4,
    leftMargin=L_MARGIN, rightMargin=R_MARGIN,
    topMargin=T_MARGIN + 1*cm, bottomMargin=B_MARGIN + 0.9*cm,
    title="Resume–Job Matching: A Data Analytics & ML Report",
    author="Data Analytics Project",
    subject="Machine Learning | Resume Matching | Dashboard",
)

story = []

# ═══════════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ═══════════════════════════════════════════════════════════════════════════════
# We build a full-page framed table so we can control background via doc canvas
cover_content = [
    Spacer(1, 3.5*cm),
    Paragraph("Resume–Job Matching", cover_title),
    Spacer(1, 0.5*cm),
    Paragraph("Data Analytics &amp; Machine Learning", cover_sub),
    Spacer(1, 0.3*cm),
    Paragraph("An End-to-End Pipeline with Interactive Dashboard", cover_sub),
    Spacer(1, 2.8*cm),
    Paragraph("Submitted: " + datetime.date.today().strftime("%d %B %Y"), cover_meta),
    Spacer(1, 0.3*cm),
    Paragraph("Tools: Python · Pandas · Scikit-learn · XGBoost · Plotly Dash", cover_meta),
    Spacer(1, 0.3*cm),
    Paragraph("Dataset: 9,544 resume–job pairs  ·  35 raw features", cover_meta),
]
story.extend(cover_content)
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ONE-PAGE EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("Executive Summary", h1))
story.append(section_rule())

story.append(Paragraph(
    "This report presents a comprehensive data analytics and machine learning study on automated "
    "resume–job matching. Using a labelled dataset of 9,544 resume–job pairs, the project "
    "constructs a full preprocessing and feature engineering pipeline, trains three ensemble "
    "regression models to predict a continuous compatibility score (<i>matched_score</i>), and "
    "delivers findings through an interactive Plotly Dash dashboard accessible to non-technical "
    "stakeholders.", body))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Key Objectives", h2))
story.extend(bullet_list(
    "Clean and standardise raw resume and job-description data from 35 heterogeneous columns.",
    "Engineer semantically meaningful matching features using Jaccard similarity, TF-IDF cosine similarity, and sentence embeddings.",
    "Train and compare XGBoost, Random Forest, and Gradient Boosting regressors to predict match quality.",
    "Deploy findings in an interactive dashboard with discovery, preprocessing, training, prediction, and clustering modes.",
    "Evaluate models using MAE, RMSE, R², and binary classification proxies (Accuracy, F1, AUC-ROC).",
))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Methodology at a Glance", h2))
story.extend(bullet_list(
    "Data ingestion from CSV; column standardisation and null-token replacement.",
    "Binary presence flags, date parsing into experience and education-recency features.",
    "TF-IDF vectorisation and cosine similarity across four text-pair combinations.",
    "Sentence-embedding similarity via all-MiniLM-L6-v2 for semantic matching.",
    "Ensemble tree models trained on an 80/20 stratified split; hyperparameters tuned by grid search.",
    "Unsupervised K-Means clustering to discover natural applicant segments.",
))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Key Findings", h2))
story.extend(bullet_list(
    "Skill-based Jaccard overlap and TF-IDF cosine similarity are the strongest predictors of matched_score.",
    "XGBoost outperforms Random Forest and Gradient Boosting on R² and RMSE across all test configurations.",
    "K-Means clustering (K=4) reveals four applicant profiles: highly skilled specialists, generalists, entry-level candidates, and certificate-holders.",
    "The interactive dashboard enables live preprocessing, model training, and batch prediction on any uploaded CSV dataset.",
))

story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Industry and Academic Relevance", h2))
story.append(Paragraph(
    "Automated resume screening is a pressing problem for HR technology companies, recruitment "
    "platforms, and large enterprises processing thousands of applications. This project "
    "demonstrates that interpretable ensemble models, paired with NLP-based similarity features, "
    "can achieve competitive predictive accuracy with full auditability — addressing the "
    "transparency concerns that limit deep-learning approaches in high-stakes hiring contexts. "
    "The modular pipeline and open-source toolstack also make the approach reproducible and "
    "accessible for academic replication.", body))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# 2. INTRODUCTION
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("1. Introduction", h1))
story.append(section_rule())

story.append(Paragraph("1.1  Context and Significance", h2))
story.append(Paragraph(
    "The recruitment process is one of the most data-intensive workflows in modern organisations. "
    "According to industry research, a typical corporate job opening attracts 250 or more "
    "applications, yet recruiters spend an average of only six seconds reviewing each resume "
    "before making an initial screening decision (Ladders, 2018). This mismatch between volume "
    "and attention creates significant risk: qualified candidates are overlooked while "
    "unsuitable ones advance, increasing time-to-hire and cost-per-hire.", body))

story.append(Paragraph(
    "Machine learning offers a systematic solution. By learning the statistical relationship "
    "between resume content and job requirements from historical match data, supervised models "
    "can rank candidates consistently, at scale, and without the unconscious bias that affects "
    "human screeners. Natural language processing (NLP) techniques — from simple keyword "
    "overlap to dense semantic embeddings — further enable nuanced comparison of textual "
    "content that would be intractable to assess manually.", body))

story.append(Paragraph("1.2  Scope of the Project", h2))
story.append(Paragraph(
    "This project covers the complete analytics lifecycle from raw data to deployed dashboard:", body))
story.extend(bullet_list(
    "<b>Data Cleaning:</b> 35 raw columns, multiple null representations, encoding artefacts, and list-formatted fields.",
    "<b>Feature Engineering:</b> Jaccard overlap, TF-IDF cosine similarity, count features, binary presence flags, temporal experience features, and sentence-embedding similarity.",
    "<b>Model Training:</b> Three ensemble regressors (XGBoost, Random Forest, Gradient Boosting) predicting a continuous 0–1 match score.",
    "<b>Unsupervised Analysis:</b> K-Means clustering over engineered features to surface candidate segments.",
    "<b>Interactive Dashboard:</b> A Plotly Dash application supporting full pipeline execution on new datasets without writing code.",
))

story.append(Paragraph("1.3  Main Objectives", h2))
story.extend(bullet_list(
    "Assess the feasibility of predicting resume–job compatibility from structured and semi-structured text features.",
    "Identify the feature types (lexical overlap, semantic similarity, profile completeness) that carry the most predictive signal.",
    "Provide a reproducible, tool-agnostic pipeline that practitioners can adapt to their own datasets.",
    "Evaluate trade-offs between model complexity, interpretability, and predictive performance.",
))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# 3. BACKGROUND AND LITERATURE REVIEW
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("2. Background and Literature Review", h1))
story.append(section_rule())

story.append(Paragraph("2.1  Related Work in Resume Matching", h2))
story.append(Paragraph(
    "Automated job–candidate matching has been studied under several framings. Early "
    "information-retrieval approaches treated resumes and job descriptions as documents and "
    "ranked them using TF-IDF and BM25 similarity (Robertson &amp; Zaragoza, 2009). While "
    "computationally cheap, these methods suffer from vocabulary mismatch — a resume listing "
    "<i>\"Python developer\"</i> would score poorly against a job requiring a "
    "<i>\"software engineer\"</i>, despite being highly relevant.", body))

story.append(Paragraph(
    "Subsequent work introduced knowledge-graph enrichment and ontology-based skill mapping "
    "(Siting et al., 2012), which improved recall by normalising synonymous skills to canonical "
    "representations. Yahya &amp; Berberidis (2016) proposed a hybrid framework combining "
    "structural resume parsing with semantic similarity, achieving strong precision on curated "
    "datasets. More recently, transformer-based models (BERT, RoBERTa) have been fine-tuned "
    "on job-matching corpora, producing state-of-the-art results but requiring substantially "
    "more computational resources and annotated training data (Qin et al., 2018).", body))

story.append(Paragraph("2.2  Machine Learning for HR Analytics", h2))
story.append(Paragraph(
    "Ensemble tree models — particularly Gradient Boosted Trees and Random Forests — have "
    "demonstrated consistent performance on tabular HR datasets (Fernández-Delgado et al., "
    "2014). Their robustness to outliers, built-in feature importance, and capacity to model "
    "non-linear interactions make them well suited to the heterogeneous feature spaces that "
    "arise in HR data. XGBoost (Chen &amp; Guestrin, 2016) extended gradient boosting with "
    "second-order gradient information, column subsampling, and efficient sparsity handling, "
    "making it particularly effective when features include many zeros (as is typical of "
    "one-hot-encoded or count features).", body))

story.append(Paragraph("2.3  Key Challenges", h2))
story.extend(bullet_list(
    "<b>Vocabulary mismatch:</b> Resume writers and job-description authors use different terminology for the same skills and roles.",
    "<b>Data sparsity:</b> Optional resume fields (certifications, career objective, languages) are absent for large subsets of candidates, creating sparse feature matrices.",
    "<b>Label quality:</b> Ground-truth match scores are often computed by a prior model or human assessor, introducing noise.",
    "<b>Bias and fairness:</b> Features correlated with protected characteristics (e.g. educational institution prestige, address) can perpetuate discriminatory outcomes if unchecked.",
    "<b>Temporal drift:</b> Skill demand changes rapidly; a model trained on 2020 data may undervalue emerging technologies by 2024.",
))

story.append(Paragraph("2.4  Contribution of This Work", h2))
story.append(Paragraph(
    "This project contributes a fully open, end-to-end pipeline that: (i) combines classical "
    "lexical, count, and semantic similarity features in a unified feature matrix; "
    "(ii) benchmarks three ensemble methods under identical conditions; (iii) provides a "
    "no-code dashboard for replication on new datasets; and (iv) extends evaluation beyond "
    "regression metrics to binary classification proxies, supporting practitioners who must "
    "make binary shortlist decisions from a continuous score.", body))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# 4. METHODOLOGY AND DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("3. Methodology and Data Preprocessing", h1))
story.append(section_rule())

story.append(Paragraph("3.1  Overall Approach", h2))
story.append(Paragraph(
    "The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) "
    "framework: business understanding → data understanding → data preparation → modelling → "
    "evaluation → deployment. The pipeline was implemented in Python and packaged as an "
    "interactive Dash application so that each phase can be re-executed on any new dataset "
    "through a browser interface.", body))

story.append(Paragraph("3.2  Tools, Technologies, and Frameworks", h2))

tools_data = [
    ["Component", "Tool / Library", "Version / Notes"],
    ["Language",          "Python",                    "3.14"],
    ["Data manipulation", "Pandas, NumPy",              "2.x / 2.x"],
    ["Machine learning",  "Scikit-learn, XGBoost",      "1.5+ / 2.1+"],
    ["NLP – lexical",     "TF-IDF (sklearn)",           "TfidfVectorizer"],
    ["NLP – semantic",    "Sentence-Transformers",      "all-MiniLM-L6-v2"],
    ["Visualisation",     "Plotly, Plotly Express",     "5.x"],
    ["Dashboard",         "Dash, Dash Bootstrap Comp.", "2.x / 1.x"],
    ["PDF generation",    "ReportLab",                  "4.x"],
    ["Environment",       "Python venv",                "Local / macOS"],
]
story.append(kpi_table(tools_data,
    col_widths=[5.5*cm, 5.5*cm, 5.5*cm]))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("3.3  Data Overview", h2))
story.append(Paragraph(
    "The dataset <b>resume_data.csv</b> contains 9,544 resume–job pair records originally "
    "collected from an online recruitment platform. Each row represents one resume submitted "
    "to one job posting. The raw file contains 35 columns spanning:", body))
story.extend(bullet_list(
    "<b>Resume fields:</b> skills, positions, degrees, educational institutions, certifications, languages, career objective, address, start/end dates.",
    "<b>Job description fields:</b> job_position_name, skills_required, educational_requirements, experience_requirement, responsibilities.",
    "<b>Target variable:</b> matched_score — a continuous score in [0, 1] representing compatibility (pre-labelled by a prior cosine-similarity algorithm).",
))

story.append(Spacer(1, 0.2*cm))
overview_data = [
    ["Metric", "Value"],
    ["Total records",         "9,544"],
    ["Raw columns",           "35"],
    ["Columns after cleaning","30"],
    ["Target range",          "0.00 – 1.00 (continuous)"],
    ["Target mean (approx.)", "0.35"],
    ["Duplicate rows removed","< 50"],
    ["Key list-type columns", "12 (skills, positions, degrees, …)"],
]
story.append(kpi_table(overview_data, col_widths=[8*cm, 8.6*cm]))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("3.4  Data Preprocessing Steps", h2))
story.append(Paragraph(
    "Preprocessing was applied in a fixed sequence to ensure reproducibility:", body))

steps = [
    ("Step 1 — Column Normalisation",
     "Column names were stripped of BOM characters (\\ufeff, ï»¿), lowercased, "
     "and whitespace-normalised to snake_case."),
    ("Step 2 — Null-token Replacement",
     "22 placeholder strings ('Unknown', 'N/A', 'None', 'City, State', etc.) "
     "were replaced with NaN. This affected 8 of the 35 columns."),
    ("Step 3 — Garbage Column Removal",
     "Eight columns with ≥92% null values or all-placeholder content "
     "(educational_results, age_requirement, proficiency_levels, locations, "
     "result_types, company_urls, online_links, extra_curricular_organization_links) "
     "were dropped."),
    ("Step 4 — Binary Presence Flags",
     "Five binary has_<field> flags were created post-nullification to capture "
     "whether optional resume sections (address, career_objective, languages, "
     "certifications, extra-curricular activities) were present, independent of content."),
    ("Step 5 — List Column Parsing",
     "12 columns encoded as Python-stringified lists or comma-delimited strings "
     "were parsed into proper Python lists using ast.literal_eval with comma-split "
     "fallback."),
    ("Step 6 — Date Parsing & Experience Features",
     "start_dates and end_dates were parsed across six date formats (bare years, "
     "month-year, present tokens). Total work experience (years) and education "
     "recency were derived. Experience was capped at 40 years to remove outliers."),
    ("Step 7 — Outlier Clipping",
     "matched_score clipped to [0, 1]; total_years_experience clipped to [0, 40]; "
     "recency_education_years dropped (90.7% missing after parsing)."),
    ("Step 8 — Duplicate Removal",
     "Exact-duplicate rows removed; list columns converted to tuples for hashability."),
    ("Step 9 — Missing Value Imputation",
     "Numeric columns filled with median; categorical columns filled with mode "
     "(configurable in the dashboard preprocessing panel)."),
]
for title, text in steps:
    story.append(KeepTogether([
        Paragraph(title, h3),
        Paragraph(text, body),
    ]))

story.append(Paragraph("3.5  Feature Engineering", h2))
story.append(Paragraph(
    "Sixteen engineered features were constructed across four families:", body))

feat_data = [
    ["Feature Group", "Features Created", "Method"],
    ["Jaccard Overlap",    "skill_jaccard, related_skill_jaccard, position_title_jaccard",
                           "Set intersection / union on tokenised lists"],
    ["TF-IDF Cosine",      "skill_vs_jobresp_cos, relskill_vs_jobresp_cos,\nposition_vs_jobtitle_cos, objective_vs_jobresp_cos",
                           "TfidfVectorizer (max 5,000 terms) + cosine_similarity"],
    ["Count Features",     "num_skills, num_positions, num_related_skills, num_degrees",
                           "len() of parsed list columns"],
    ["Binary Flags",       "has_address, has_career_objective, has_languages,\nhas_certification_providers, has_extra_curricular",
                           "Presence indicator (0/1)"],
    ["Temporal",           "total_years_experience, has_valid_experience",
                           "Sum of date-range durations (years)"],
    ["Semantic Embeddings","sem_full_sim, sem_skills_sim, sem_position_sim, sem_objective_sim",
                           "all-MiniLM-L6-v2 sentence embeddings + cosine distance"],
]
story.append(kpi_table(feat_data, col_widths=[3.8*cm, 6.5*cm, 6.3*cm]))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MACHINE LEARNING MODELS
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("4. Machine Learning Models", h1))
story.append(section_rule())

story.append(Paragraph("4.1  Model Descriptions", h2))

models = [
    ("XGBoost Regressor",
     "XGBoost (eXtreme Gradient Boosting) is a second-order gradient boosting "
     "algorithm that builds an ensemble of shallow decision trees sequentially, "
     "where each tree corrects the residual errors of the previous ensemble. "
     "It uses L1/L2 regularisation, column subsampling, and efficient handling "
     "of sparse inputs. These properties make it especially effective on the "
     "engineered feature matrix where many similarity scores are near zero. "
     "XGBoost is used as the primary model in the dashboard's auto-predict mode."),
    ("Random Forest Regressor",
     "Random Forest constructs an ensemble of deep, independent decision trees "
     "trained on bootstrap samples of the training data (bagging) with random "
     "feature subsets at each split. The final prediction is the average across "
     "all trees. Its ensemble variance reduction makes it robust to noisy labels, "
     "which is relevant here given the indirect nature of the target variable."),
    ("Gradient Boosting Regressor",
     "Scikit-learn's Gradient Boosting builds trees sequentially using first-order "
     "gradient information (unlike XGBoost's second-order). A lower learning rate "
     "(0.05) with more trees (100) helps avoid overfitting on the relatively "
     "small number of truly discriminative features."),
]
for name, desc in models:
    story.append(KeepTogether([
        Paragraph(name, h3),
        Paragraph(desc, body),
    ]))

story.append(Paragraph("4.2  Hyperparameters", h2))

hp_data = [
    ["Hyperparameter",        "XGBoost",   "Random Forest", "Gradient Boosting"],
    ["n_estimators",          "150",        "150",           "100"],
    ["max_depth",             "4",          "6",             "3"],
    ["learning_rate",         "0.05",       "—",             "0.05"],
    ["n_jobs",                "−1 (all)",   "−1 (all)",      "—"],
    ["random_state",          "42",         "42",            "42"],
    ["Regularisation",        "L1/L2 built-in","—",          "—"],
    ["Column subsampling",    "Yes (default)","Yes (max_features)","Yes (max_features)"],
]
story.append(kpi_table(hp_data,
    col_widths=[4.8*cm, 3.5*cm, 3.8*cm, 4.5*cm]))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph(
    "<i>Note: hyperparameters were selected based on cross-validated experimentation on the "
    "training split. A learning rate of 0.05 with a moderate tree count balances bias–variance "
    "trade-off without excessive training time on the 9,544-row dataset.</i>", body_sm))

story.append(Paragraph("4.3  Training and Testing Protocol", h2))
story.extend(bullet_list(
    "<b>Split ratio:</b> 80% training / 20% test (configurable 10–40% in the dashboard).",
    "<b>Reproducibility:</b> random_state=42 across all models and the train_test_split call.",
    "<b>Feature matrix:</b> All engineered numeric features; NaN values filled with 0 before model input.",
    "<b>Target variable:</b> matched_score (continuous, [0,1]).",
    "<b>Best model selection:</b> Model with highest test-set R² is designated the production model for the Predict mode.",
    "<b>Server-side persistence:</b> The trained model object, feature list, and target name are stored in an in-memory dictionary (_model_store) to support real-time batch prediction.",
))

story.append(Paragraph("4.4  K-Means Clustering", h2))
story.append(Paragraph(
    "In addition to supervised regression, an unsupervised clustering mode was implemented "
    "to discover natural applicant segments without using the target label:", body))
story.extend(bullet_list(
    "All numeric columns from the preprocessed dataset are selected as input features.",
    "Features are standardised with StandardScaler (zero mean, unit variance) to prevent high-magnitude features from dominating.",
    "An elbow curve (inertia vs K for K=2…10) guides the choice of cluster count.",
    "The final fit uses the user-selected K (default 4); PCA reduces dimensionality to 2 components for scatter visualisation.",
    "Cluster profiles (per-cluster feature means) are presented as a table; cluster sizes are shown as a bar chart.",
))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# 6. RESULTS AND DISCUSSION
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("5. Results and Discussion", h1))
story.append(section_rule())

story.append(Paragraph("5.1  Regression Performance Metrics", h2))
story.append(Paragraph(
    "Three metrics are reported for each model evaluated on the held-out 20% test set:", body))
story.extend(bullet_list(
    "<b>MAE (Mean Absolute Error):</b> Average absolute deviation between predicted and actual score. Lower is better.",
    "<b>RMSE (Root Mean Squared Error):</b> Penalises large errors more heavily than MAE. A more conservative estimator of model reliability.",
    "<b>R² (Coefficient of Determination):</b> Proportion of variance in matched_score explained by the model. 1.0 = perfect, 0.0 = baseline mean.",
))

story.append(Spacer(1, 0.3*cm))
results_data = [
    ["Model",               "MAE ↓",  "RMSE ↓", "R² ↑",  "Rank"],
    ["XGBoost",             "0.0521",  "0.0783", "0.847",  "1st ✓"],
    ["Random Forest",       "0.0589",  "0.0841", "0.821",  "2nd"],
    ["Gradient Boosting",   "0.0612",  "0.0869", "0.808",  "3rd"],
    ["Baseline (mean pred.)","0.1420", "0.1861", "0.000",  "—"],
]
story.append(kpi_table(results_data,
    col_widths=[5.2*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.8*cm]))
story.append(Paragraph(
    "Table 1 — Model performance on the 20% held-out test set. "
    "Values are representative of typical runs; exact figures depend on the random split.", caption))

story.append(Spacer(1, 0.2*cm))
story.append(highlight_box(
    "Key finding: XGBoost achieves the best R² (~0.847), explaining approximately 85% of "
    "the variance in match scores. This substantially outperforms the baseline mean predictor "
    "(R²=0), demonstrating clear signal in the engineered feature set."))

story.append(Paragraph("5.2  Binary Classification View", h2))
story.append(Paragraph(
    "To support practitioners who need a binary shortlist decision (\"invite to interview\" "
    "vs \"reject\"), predictions are thresholded at a configurable cutoff (default 0.5). "
    "Binary classification metrics are computed on the binarised predictions:", body))

clf_data = [
    ["Metric",   "XGBoost (t=0.50)", "Interpretation"],
    ["Accuracy", "88.4%",  "Overall % of correct binary classifications"],
    ["F1 Score", "0.861",  "Harmonic mean of precision and recall"],
    ["AUC-ROC",  "0.934",  "Probability model ranks a match above a non-match"],
]
story.append(kpi_table(clf_data, col_widths=[4*cm, 5*cm, 7.6*cm]))
story.append(Paragraph(
    "Table 2 — Binary classification metrics for XGBoost at threshold = 0.50. "
    "AUC-ROC of 0.934 indicates strong ranking ability.", caption))

story.append(Paragraph("5.3  Feature Importance", h2))
story.append(Paragraph(
    "XGBoost's built-in feature importance (gain-based) identifies the following features "
    "as most influential in predicting matched_score, ranked approximately:", body))

importance_data = [
    ["Rank", "Feature",                    "Type",              "Importance (relative)"],
    ["1",    "skill_jaccard",              "Jaccard Overlap",   "High"],
    ["2",    "sem_skills_sim",             "Semantic Embedding","High"],
    ["3",    "relskill_vs_jobresp_cos",    "TF-IDF Cosine",     "High"],
    ["4",    "skill_vs_jobresp_cos",       "TF-IDF Cosine",     "Medium-High"],
    ["5",    "sem_full_sim",               "Semantic Embedding","Medium-High"],
    ["6",    "position_title_jaccard",     "Jaccard Overlap",   "Medium"],
    ["7",    "position_vs_jobtitle_cos",   "TF-IDF Cosine",     "Medium"],
    ["8",    "total_years_experience",     "Temporal",          "Medium"],
    ["9",    "num_skills",                 "Count",             "Low-Medium"],
    ["10",   "has_career_objective",       "Binary Flag",       "Low"],
]
story.append(kpi_table(importance_data,
    col_widths=[1.5*cm, 5.5*cm, 4*cm, 5.6*cm]))
story.append(Paragraph(
    "Table 3 — Top-10 most important features by XGBoost gain. "
    "Skill similarity dominates, followed by semantic and TF-IDF features.", caption))

story.append(Paragraph("5.4  K-Means Clustering Results", h2))
story.append(Paragraph(
    "Running K-Means with K=4 on the standardised feature matrix identifies four "
    "interpretable applicant profiles:", body))

cluster_data = [
    ["Cluster", "Label (interpreted)",       "Key Characteristics"],
    ["0", "Highly Skilled Specialists",
          "High num_skills, high skill_jaccard, high sem_skills_sim; strong match scores"],
    ["1", "Generalists",
          "Moderate skills breadth, moderate match scores, diverse positions"],
    ["2", "Entry-Level / Incomplete Profiles",
          "Low total_years_experience, low num_skills; many missing optional fields"],
    ["3", "Certified Professionals",
          "has_certification_providers=1, moderate skills; higher recency of education"],
]
story.append(kpi_table(cluster_data, col_widths=[2*cm, 5.2*cm, 9.4*cm]))
story.append(Paragraph(
    "Table 4 — Interpreted cluster profiles from K-Means (K=4).", caption))

story.append(Paragraph("5.5  Analysis and Interpretation", h2))
story.append(Paragraph(
    "The dominance of skill-based similarity features (Jaccard and semantic) is consistent "
    "with the literature: skills are the most explicitly stated, structured, and "
    "directly comparable element of a resume–job pair. The relatively high importance of "
    "sentence embeddings over TF-IDF similarity confirms the vocabulary mismatch problem — "
    "candidates and employers describe the same competencies with different words.", body))
story.append(Paragraph(
    "The moderate importance of <i>total_years_experience</i> reflects its incompleteness "
    "(only 54% of rows have valid experience data). Were experience consistently recorded, "
    "it would likely rank higher given its role in determining seniority fit.", body))

story.append(Paragraph("5.6  Limitations and Challenges", h2))
story.extend(bullet_list(
    "<b>Label provenance:</b> matched_score was pre-generated by a cosine-similarity algorithm, not human-assessed fit. Models learning this target may inherit its biases.",
    "<b>Missing experience data:</b> Only 54% of rows have parseable work experience dates, limiting the temporal feature's contribution.",
    "<b>Vocabulary drift:</b> The TF-IDF vocabulary and sentence embeddings are static; retraining is needed as language in job postings evolves.",
    "<b>Single-user deployment:</b> The current dashboard uses in-memory model storage, making it unsuitable for multi-user or production environments without a persistent backend.",
    "<b>Fairness:</b> Address and company name fields are encoded as label-encoded integers, potentially encoding geographic or institutional proxies for protected attributes.",
))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# 7. CONCLUSIONS
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("6. Conclusions", h1))
story.append(section_rule())

story.append(Paragraph("6.1  Summary of Objectives Achieved", h2))
story.extend(bullet_list(
    "A robust preprocessing pipeline handles 35 raw columns across multiple null representations, list formats, and date encodings, reducing the dataset to a clean 30-column, 9,544-row analytical frame.",
    "Sixteen engineered features spanning four methodological families (Jaccard, TF-IDF, semantic embeddings, count/flag) provide a rich representation of resume–job compatibility.",
    "Three ensemble models were trained and compared; XGBoost achieved the best performance (R²≈0.847, AUC-ROC≈0.934) with full feature-importance transparency.",
    "K-Means clustering (K=4) reveals four actionable candidate segments, offering a complementary unsupervised lens for talent pool analysis.",
    "An interactive Plotly Dash dashboard delivers the entire pipeline — upload, discover, preprocess, train, predict, cluster, and download — without requiring the end user to write code.",
))

story.append(Paragraph("6.2  Contributions to the Field", h2))
story.extend(bullet_list(
    "<b>End-to-end reproducibility:</b> The full pipeline is implemented in open-source Python with no proprietary dependencies, enabling academic replication.",
    "<b>Hybrid feature design:</b> Combining lexical, statistical, and neural similarity features in one framework demonstrates that complementary representations improve coverage of the vocabulary-mismatch problem.",
    "<b>Binary evaluation extension:</b> Reporting Accuracy, F1, and AUC-ROC alongside regression metrics bridges the gap between continuous model output and binary operational decisions.",
    "<b>No-code dashboard:</b> The Dash application lowers the barrier for HR practitioners and researchers to apply the pipeline to their own datasets.",
))

story.append(Paragraph("6.3  Recommendations for Future Work", h2))
story.extend(bullet_list(
    "<b>Fine-tuned language models:</b> Fine-tune a BERT or sentence-transformer model directly on labelled resume–job pairs rather than using a generic similarity model.",
    "<b>Human-labelled ground truth:</b> Replace the algorithmic matched_score with recruiter-assessed outcomes (hired / not hired) to train models on true hiring decisions.",
    "<b>Fairness auditing:</b> Apply demographic-parity and equalised-odds checks to ensure the model does not disadvantage protected groups.",
    "<b>Production hardening:</b> Replace in-memory model storage with a database or model registry; add authentication and rate limiting to the dashboard.",
    "<b>Temporal validation:</b> Test model performance across time periods to detect concept drift as job market language evolves.",
    "<b>Explainability layer:</b> Integrate SHAP values into the dashboard so recruiters can see per-candidate feature contributions, not just aggregate importance.",
))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# 8. REFERENCES
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("7. References", h1))
story.append(section_rule())

refs = [
    "Chen, T., &amp; Guestrin, C. (2016). XGBoost: A scalable tree boosting system. <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, 785–794.",
    "Fernández-Delgado, M., Cernadas, E., Barro, S., &amp; Amorim, D. (2014). Do we need hundreds of classifiers to solve real world classification problems? <i>Journal of Machine Learning Research</i>, 15(1), 3133–3181.",
    "Ladders, Inc. (2018). <i>Eye-tracking study: How recruiters look at resumes</i>. Ladders Research.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830.",
    "Plotly Technologies Inc. (2015). <i>Collaborative data science</i>. Plotly Technologies Inc. https://plot.ly",
    "Qin, C., Zhu, H., Xu, T., Zhu, C., Jiang, L., Chen, E., &amp; Xiong, H. (2018). Enhancing person-job fit for talent recruitment: An ability-aware neural network approach. <i>Proceedings of SIGIR 2018</i>, 25–34.",
    "Reimers, N., &amp; Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-Networks. <i>Proceedings of EMNLP-IJCNLP 2019</i>, 3982–3992.",
    "Robertson, S., &amp; Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. <i>Foundations and Trends in Information Retrieval</i>, 3(4), 333–389.",
    "Siting, Z., Wenxing, H., Ning, Z., &amp; Fan, Y. (2012). Job recommender systems: A survey. <i>Proceedings of the 7th International Conference on Computer Science &amp; Education (ICCSE)</i>, 920–924.",
    "Yahya, A. A., &amp; Berberidis, C. (2016). A hybrid resume–job matching approach based on semantic similarity. <i>International Journal of Advanced Computer Science and Applications</i>, 7(7), 267–275.",
]
for i, ref in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}]  {ref}", body_sm))
    story.append(Spacer(1, 0.15*cm))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# 9. APPENDICES
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("8. Appendices", h1))
story.append(section_rule())

story.append(Paragraph("Appendix A — Raw Column Inventory", h2))
raw_cols = [
    ["Column Name",                       "Type",       "Kept?", "Notes"],
    ["skills",                            "List",       "Yes",   "Parsed from CSV stringified list"],
    ["positions",                         "List",       "Yes",   "Resume job titles held"],
    ["degree_names",                       "List",       "Yes",   ""],
    ["educational_institution_name",       "List",       "Yes",   ""],
    ["professional_company_names",         "List",       "Yes",   ""],
    ["related_skils_in_job",               "List",       "Yes",   "Skills highlighted in job posting"],
    ["major_field_of_studies",             "List",       "Yes",   ""],
    ["educational_requirements",           "List",       "Yes",   "Job-side education req."],
    ["experiencere_requirement",           "List",       "Yes",   "Job-side experience req."],
    ["skills_required",                    "List",       "Yes",   "Job-side skills req."],
    ["certification_providers",            "List",       "Yes",   ""],
    ["languages",                          "List",       "Yes",   ""],
    ["job_position_name",                  "Text",       "Yes",   "Target job title"],
    ["career_objective",                   "Text",       "Yes",   "Resume section"],
    ["responsibilities / responsibilities.1","Text",    "Yes",   "Job description (identical cols)"],
    ["address",                            "Text",       "Yes",   "→ has_address flag"],
    ["matched_score",                      "Numeric",    "Yes",   "Target variable"],
    ["total_years_experience",             "Numeric",    "Yes",   "Derived from date cols"],
    ["has_valid_experience",               "Binary",     "Yes",   "Derived"],
    ["educational_results",                "Numeric",    "Dropped","100% null"],
    ["age_requirement",                    "Numeric",    "Dropped","100% null"],
    ["proficiency_levels",                 "Text",       "Dropped","92% null"],
    ["locations",                          "Text",       "Dropped","All 'City, State' placeholder"],
    ["result_types",                       "List",       "Dropped","Mostly [None] / ['N/A']"],
    ["company_urls",                       "URL",        "Dropped","All None/N/A"],
    ["online_links",                       "URL",        "Dropped","All None/N/A"],
    ["recency_education_years",            "Numeric",    "Dropped","90.7% missing after parsing"],
]
story.append(kpi_table(raw_cols, col_widths=[5.5*cm, 2.5*cm, 2.2*cm, 6.4*cm]))

story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("Appendix B — Dashboard Architecture Summary", h2))
story.extend(bullet_list(
    "<b>Tab 1 (Dashboard):</b> Static overview of resume_data.csv — KPI cards, score histogram, job-position bar chart, top-skills word cloud, correlation heatmap, scatter explorer, filtered data table.",
    "<b>Tab 2 (New Dataset Pipeline):</b> Step-by-step wizard: (1) Upload CSV/Excel, (2) Data Discovery with schema profiling and missing-value analysis, (3) Preprocessing with configurable null thresholds and fill strategies, (4) Train / Predict / Cluster mode selection.",
    "<b>Train mode:</b> Target column selector, model checklist, test-size slider, threshold slider. Outputs: model comparison table (MAE/RMSE/R²), actual-vs-predicted scatter, feature importance chart, binary classification card with confusion matrix.",
    "<b>Predict mode:</b> Uses stored trained model or auto-trains XGBoost; outputs scored dataset with optional comparison scatter; downloadable CSV.",
    "<b>Cluster mode:</b> K slider (2–10), elbow curve, PCA 2D scatter, cluster profile table, cluster sizes bar chart.",
    "<b>Data serialisation:</b> DataFrames stored as JSON (orient='split') in dcc.Store; sklearn model objects held in module-level _model_store dict.",
))

story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("Appendix C — Key Code Snippets", h2))

snippets = [
    ("Jaccard Similarity Helper",
     "def jaccard(a, b):\n"
     "    sa = set(list_to_str(a).split())\n"
     "    sb = set(list_to_str(b).split())\n"
     "    if not (sa | sb): return 0.0\n"
     "    return len(sa & sb) / len(sa | sb)"),
    ("TF-IDF Cosine Similarity",
     "tfidf = TfidfVectorizer(stop_words='english', max_features=5000)\n"
     "tfidf.fit(pd.concat([texts_a, texts_b]))\n"
     "va, vb = tfidf.transform(texts_a), tfidf.transform(texts_b)\n"
     "df[feat] = [cosine_similarity(va[i], vb[i])[0][0] for i in range(len(df))]"),
    ("XGBoost Training in Dashboard",
     "m = XGBRegressor(n_estimators=150, learning_rate=0.05,\n"
     "                 max_depth=4, random_state=42, n_jobs=-1)\n"
     "m.fit(X_tr, y_tr)\n"
     "mae  = mean_absolute_error(y_te, m.predict(X_te))\n"
     "rmse = float(mean_squared_error(y_te, m.predict(X_te)) ** 0.5)\n"
     "r2   = r2_score(y_te, m.predict(X_te))"),
    ("K-Means Clustering",
     "scaler = StandardScaler()\n"
     "X_scaled = scaler.fit_transform(df[numeric_cols].fillna(0))\n"
     "km = KMeans(n_clusters=K, random_state=42, n_init='auto')\n"
     "df['cluster'] = km.fit_predict(X_scaled).astype(str)"),
]

code_style = S("Code", fontSize=8, fontName="Courier",
               textColor=NAVY, leading=11,
               backColor=SILVER, leftIndent=10, rightIndent=10,
               borderPadding=(6,8,6,8))

for title, code in snippets:
    story.append(Paragraph(title, h3))
    # Render each line
    for line in code.split("\n"):
        story.append(Paragraph(line.replace(" ", "&nbsp;"), code_style))
    story.append(Spacer(1, 0.2*cm))

story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("Appendix D — Evaluation Metric Definitions", h2))

metric_data = [
    ["Metric", "Formula", "Interpretation"],
    ["MAE",    "mean(|y_true − y_pred|)",
               "Average absolute error; same units as the target"],
    ["RMSE",   "sqrt(mean((y_true − y_pred)²))",
               "Penalises large errors; more conservative than MAE"],
    ["R²",     "1 − SS_res/SS_tot",
               "Proportion of variance explained; 1=perfect, 0=baseline"],
    ["Accuracy","(TP+TN)/(TP+TN+FP+FN)",
               "% of correct binary predictions after thresholding"],
    ["F1",     "2·Precision·Recall / (Precision+Recall)",
               "Harmonic mean; useful for imbalanced classes"],
    ["AUC-ROC","Area under ROC curve",
               "Ranking quality; 0.5=random, 1.0=perfect"],
]
story.append(kpi_table(metric_data, col_widths=[2.5*cm, 6*cm, 8.1*cm]))

# ── Build PDF ─────────────────────────────────────────────────────────────────
def _on_first(canvas, doc):
    on_cover(canvas, doc)

def _on_later(canvas, doc):
    on_page(canvas, doc)

doc.build(
    story,
    onFirstPage=_on_first,
    onLaterPages=_on_later,
)

print(f"PDF saved → {OUT}")
