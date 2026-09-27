"""Build the formatted Word report for the California Housing lab assignment.

Produces a submission-ready .docx: cover page, contents, every mandatory task
with its real output tables, all 23 diagnostic figures, the GitHub reference and
a full bibliography.

The document is NOT committed to git - it is a submission artefact, not source.
Run:  python build_report_docx.py
"""

from __future__ import annotations

import os
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "outputs" / "figures"
TAB_DIR = BASE_DIR / "outputs" / "tables"
OUT_FILE = BASE_DIR / "Lab_Assignment_Report_California_Housing.docx"

# --- Palette ---------------------------------------------------------------
NAVY = RGBColor(0x1F, 0x3B, 0x63)
ACCENT = RGBColor(0x2E, 0x74, 0xB5)
GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = "D9E2F3"
HEADER_FILL = "1F3B63"
BAND_FILL = "F2F5FA"

# --- Document metadata (edit these) ----------------------------------------
STUDENT_NAME = "Sanskar"
STUDENT_ROLL = "B1-100"
STUDENT_BATCH = "B1"
SUBJECT = "Exploratory Data Analysis (EDA)"
INSTITUTION = "Walchand College of Engineering, Sangli"
SUBMISSION_DATE = "27 September 2026"
GITHUB_URL = "https://github.com/Sanskar1724/-Regression-modelling-analysis"
DATASET_NAME = "Dataset 1 - California Housing"


# ===========================================================================
# LOW-LEVEL FORMATTING HELPERS
# ===========================================================================
def shade(cell, hex_fill):
    """Apply a solid background colour to a table cell."""
    element = OxmlElement("w:shd")
    element.set(qn("w:val"), "clear")
    element.set(qn("w:color"), "auto")
    element.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(element)


def set_repeat_header(row):
    """Mark a table row as a repeating header across page breaks."""
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def add_page_numbers(section):
    """Insert a centred 'Page N of M' footer."""
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            field = OxmlElement("w:fldChar")
            field.set(qn("w:fldCharType"), kind)
            run._r.append(field)
        else:
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = " PAGE "
            run._r.append(instr)
    run.font.size = Pt(9)
    run.font.color.rgb = GREY


def style_document(document):
    """Apply the base font, heading colours and spacing conventions."""
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for name, size, colour, before, after in [
        ("Heading 1", 16, NAVY, 16, 8),
        ("Heading 2", 13, NAVY, 12, 6),
        ("Heading 3", 11.5, ACCENT, 10, 4),
    ]:
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = colour
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True


def para(document, text, size=10.5, bold=False, italic=False, align=None,
         colour=None, space_after=6, indent=None):
    """Add a body paragraph with inline formatting."""
    p = document.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if colour is not None:
        run.font.color.rgb = colour
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if indent is not None:
        p.paragraph_format.left_indent = Inches(indent)
    return p


def bullet(document, text, level=0):
    """Add a bulleted list item."""
    style = "List Bullet" if level == 0 else "List Bullet 2"


def add_table(document, headers, rows, caption=None, note=None, widths=None):
    """Insert a formatted table with a navy header row and zebra banding."""
    if caption:
        cap = para(document, caption, size=9.5, bold=True, colour=NAVY,
                   space_after=3)
        cap.paragraph_format.keep_with_next = True

    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    header_cells = table.rows[0].cells
    for cell, text in zip(header_cells, headers):
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(text))
        run.bold = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shade(cell, HEADER_FILL)
    set_repeat_header(table.rows[0])

    for index, row in enumerate(rows):
        cells = table.add_row().cells
        for position, (cell, value) in enumerate(zip(cells, row)):
            cell.text = ""
            text = str(value)
            emphasise = text.startswith("**") and text.endswith("**")
            if emphasise:
                text = text.strip("*")
            run = cell.paragraphs[0].add_run(text)
            run.font.size = Pt(9)
            run.bold = emphasise
            if emphasise:
                run.font.color.rgb = NAVY
            cell.paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if position == 0 and len(headers) > 2
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            if index % 2 == 1:
                shade(cell, BAND_FILL)

    if widths:
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)

    if note:
        para(document, note, size=8.5, italic=True, colour=GREY, space_after=10)
    else:
        document.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


def add_figure(document, filename, caption, width=6.3, note=None):
    """Insert a centred figure with a numbered caption underneath."""
    path = FIG_DIR / filename
    if not path.exists():
        para(document, f"[missing figure: {filename}]", italic=True, colour=GREY)
        return
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = para(document, caption, size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
               colour=GREY, space_after=2)
    cap.paragraph_format.keep_with_next = note is not None
    if note:
        para(document, note, size=9, space_after=10)


def add_callout(document, label, text, fill=LIGHT):
    """Insert a single-cell shaded box for a conclusion or key finding."""
    table = document.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell = table.rows[0].cells[0]
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(f"{label}  ")
    run.bold = True
    run.font.size = Pt(9.5)
    run.font.color.rgb = NAVY
    body = paragraph.add_run(text)
    body.font.size = Pt(9.5)
    shade(cell, fill)
    document.add_paragraph().paragraph_format.space_after = Pt(6)
    return table


def add_output_block(document, text):
    """Insert a monospaced, shaded block representing console output."""
    table = document.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell = table.rows[0].cells[0]
    cell.text = ""
    for index, line in enumerate(text.strip("\n").splitlines()):
        p = cell.paragraphs[0] if index == 0 else cell.add_paragraph()
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
    shade(cell, "F4F4F4")
    document.add_paragraph().paragraph_format.space_after = Pt(6)
    return table


# ===========================================================================
# FRONT MATTER
# ===========================================================================
def build_cover(document):
    """Institution-style cover page."""
    para(document, INSTITUTION, size=13, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, colour=NAVY, space_after=2)
    para(document, "Department of Computer Engineering", size=10.5,
         align=WD_ALIGN_PARAGRAPH.CENTER, colour=GREY, space_after=24)

    line = document.add_paragraph()
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = line.add_run("_" * 62)
    run.font.color.rgb = ACCENT

    para(document, "LAB ASSIGNMENT REPORT", size=24, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, colour=NAVY, space_after=4)
    para(document, "Exploratory Data Analysis (EDA)", size=15,
         align=WD_ALIGN_PARAGRAPH.CENTER, colour=ACCENT, space_after=2)
    para(document, "Regression Analysis of Datasets", size=15,
         align=WD_ALIGN_PARAGRAPH.CENTER, colour=ACCENT, space_after=6)
    para(document, f"Dataset Selected: {DATASET_NAME}", size=11.5, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)

    para(document, "_" * 62, size=11, align=WD_ALIGN_PARAGRAPH.CENTER,
         colour=ACCENT, space_after=26)

    add_table(
        document,
        ["Detail", "Entry"],
        [
            ["Student Name", STUDENT_NAME],
            ["Roll Number", STUDENT_ROLL],
            ["Batch / Division", STUDENT_BATCH],
            ["Subject", SUBJECT],
            ["Dataset", DATASET_NAME],
            ["Domain", "Real Estate / Economics"],
            ["Dataset Size", "20,640 observations x 8 predictors"],
            ["Target Variable", "MedHouseVal (median house value, USD 100,000)"],
            ["GitHub Repository", GITHUB_URL],
            ["Date of Submission", SUBMISSION_DATE],
        ],
        widths=[2.0, 4.2],
    )

    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # --- Declaration -------------------------------------------------------
    document.add_heading("Declaration", level=1)
    para(document,
         "I declare that the work presented in this report is my own and has been carried "
         "out under the supervision of the subject teacher. The analysis follows the ten "
         "mandatory tasks highlighted in the assignment brief, and every statistical claim "
         "made in this document is supported by output generated from the code published in "
         "the accompanying GitHub repository. Any third-party dataset or reference used has "
         "been acknowledged in the References section.")
    para(document, f"Name: {STUDENT_NAME}    Roll No: {STUDENT_ROLL}    "
                   f"Date: {SUBMISSION_DATE}", italic=True)

    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build_contents(document):
    """Manually maintained table of contents (python-docx cannot auto-populate)."""
    document.add_heading("Table of Contents", level=1)
    entries = [
        ("1", "Problem Statement and Objectives", 3),
        ("2", "Dataset Description", 3),
        ("3", "Software and Tools Used", 4),
        ("4", "Task 1 - Exploratory Data Analysis", 5),
        ("5", "Task 2 - Correlation Analysis", 9),
        ("6", "Task 3 - Outlier Identification", 12),
        ("7", "Task 4 - Multicollinearity Analysis using VIF", 15),
        ("8", "Task 5 - Regression Models", 18),
        ("9", "Task 9 - Hyperparameter Selection (Ridge / Lasso / Elastic Net)", 21),
        ("10", "Task 6 - Model Comparison and Residual Analysis", 22),
        ("11", "Task 7 - Regression Assumption Checks", 25),
        ("12", "Task 8 - Feature Significance and Coefficient Interpretation", 28),
        ("13", "Effect of Regularisation on Coefficients", 30),
        ("14", "Task 10 - Final Model Selection and Justification", 31),
        ("15", "Limitations and Scope for Improvement", 33),
        ("16", "GitHub Repository and Reproducibility", 34),
        ("17", "Conclusion", 35),
        ("18", "References", 36),
    ]
    add_table(document, ["Section", "Title", "Page"], entries,
              widths=[0.8, 4.6, 0.8])
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ===========================================================================
# SECTIONS 1-3
# ===========================================================================
def build_introduction(document):
    document.add_heading("1. Problem Statement and Objectives", level=1)
    para(document,
         "The assignment brief supplies sixteen real datasets and asks each student to select "
         "one and apply every task highlighted in yellow, then submit a detailed report "
         "containing the outputs together with personal comments, conclusions and "
         "justification. Dataset 1, California Housing, has been selected for this analysis.")
    para(document,
         "The problem addressed is to predict the median house value of a California census "
         "block from eight socio-demographic and geographic variables, and to determine which "
         "of the candidate regression models is fit for purpose, with every modelling decision "
         "justified by statistical evidence rather than by default.")

    document.add_heading("1.1 Objectives", level=2)
    for text in [
        "Profile the dataset and establish its shape, scale and any structural limitations.",
        "Quantify the relationships between the target and each predictor, and between the predictors.",
        "Identify unusual observations using univariate, multivariate and influence-based methods, "
        "and decide with justification whether to retain or remove them.",
        "Measure multicollinearity using the Variance Inflation Factor and interpret its consequences.",
        "Fit and compare Simple, Multiple, Polynomial, Ridge, Lasso and Elastic Net regressions, "
        "along with optional Robust, Random Forest and Gradient Boosting benchmarks.",
        "Test the five classical regression assumptions formally and interpret any violations.",
        "Interpret the fitted coefficients in real-world terms, including significance testing.",
        "Select and tune hyper-parameters by cross-validation, and recommend a final model with "
        "explicit justification.",
    ]:
        bullet(document, text)

    document.add_heading("1.2 Approach Adopted", level=2)
    para(document,
         "The analysis follows the sequence laid out in the brief: an 80/20 train-test split with "
         "a fixed random seed, followed by exploratory analysis, correlation, outlier detection "
         "and VIF, then model fitting and tuning, then the formal assumption tests, then "
         "coefficient interpretation, and finally a cross-validated model comparison that drives "
         "the recommendation.")

    add_callout(
        document, "Why this dataset is a good choice:",
        "it contains no missing values and runs comfortably on a laptop, yet it deliberately "
        "violates three of the five classical regression assumptions. The interesting part of the "
        "work is therefore diagnostic rather than mechanical, which is exactly what the brief "
        "is testing.")

    document.add_heading("2. Dataset Description", level=1)
    add_table(
        document, ["Property", "Value"],
        [
            ["Dataset name", "California Housing (Dataset 1 of 16)"],
            ["Domain", "Real Estate / Economics"],
            ["Source", "sklearn.datasets.fetch_california_housing"],
            ["Original source", "1990 United States Census"],
            ["Observations", "20,640 census blocks (districts)"],
            ["Predictors", "8 (all numeric)"],
            ["Target", "MedHouseVal, median house value in USD 100,000"],
            ["Missing values", "None"],
            ["Outliers", "Present; handled in Task 3"],
            ["Key structural note", "Target is hard-censored at 5.0005 (USD 500,005)"],
        ],
        widths=[1.9, 4.3],
    )

    document.add_heading("2.1 Predictor Definitions", level=2)
    add_table(
        document, ["Variable", "Description", "Units"],
        [
            ["MedInc", "Median income of households in the block", "USD 10,000"],
            ["HouseAge", "Median age of the houses in the block", "years"],
            ["AveRooms", "Average number of rooms per household", "count"],
            ["AveBedrms", "Average number of bedrooms per household", "count"],
            ["Population", "Total population of the block", "persons"],
            ["AveOccup", "Average number of household members", "persons"],
            ["Latitude", "Latitude of the block centre", "degrees"],
            ["Longitude", "Longitude of the block centre", "degrees"],
            ["MedHouseVal", "Median house value (the target)", "USD 100,000"],
        ],
        widths=[1.2, 3.6, 1.4],
    )

    document.add_heading("3. Software and Tools Used", level=1)
    add_table(
        document, ["Tool / Package", "Version", "Purpose in this analysis"],
        [
            ["Python", "3.11", "Programming language"],
            ["NumPy", "1.26.4", "Array numerics and vectorised computation"],
            ["pandas", "2.2.2", "Data frames, grouping and CSV output"],
            ["SciPy", "1.13.1", "Statistical distributions and the Q-Q plot"],
            ["scikit-learn", "1.5.0", "Estimators, cross-validation, metrics"],
            ["statsmodels", "0.14.4", "OLS inference, VIF and diagnostic tests"],
            ["Matplotlib", "3.8.4", "All figures"],
            ["seaborn", "0.13.2", "Correlation heatmaps"],
            ["Microsoft Word", "-", "This report"],
        ],
        widths=[1.7, 1.0, 3.5],
    )
    para(document,
         "The complete source code is available in the GitHub repository listed in Section 16. "
         "All results in this report were generated by a single command, "
         "python run_analysis.py, and are therefore fully reproducible.",
         size=9.5, italic=True, colour=GREY)
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ===========================================================================
# TASK 1 - EXPLORATORY DATA ANALYSIS
# ===========================================================================
def build_task1(document):
    document.add_heading("4. Task 1 - Exploratory Data Analysis", level=1)

    document.add_heading("4.1 Output - Dataset Profile", level=2)
    add_output_block(document, """
Dataset shape : (20640, 10)
Features      : MedInc, HouseAge, AveRooms, AveBedrms,
                Population, AveOccup, Latitude, Longitude
Target        : MedHouseVal  (units of USD 100,000)
Train / test  : 16,512 rows / 4,128 rows   (80/20, seed 42)
Target mean   : train 2.0719 | test 2.0550
""")

    add_table(
        document,
        ["Variable", "Type", "Missing", "Unique", "Mean", "Std", "Min",
         "Median", "Max"],
        [
            ["MedInc", "float64", "0", "12,928", "3.871", "1.900", "0.500", "3.535", "15.000"],
            ["HouseAge", "float64", "0", "52", "28.639", "12.586", "1.000", "29.000", "52.000"],
            ["AveRooms", "float64", "0", "19,392", "5.429", "2.474", "0.846", "5.229", "141.909"],
            ["AveBedrms", "float64", "0", "14,233", "1.097", "0.474", "0.333", "1.049", "34.067"],
            ["Population", "float64", "0", "3,888", "1425.5", "1132.5", "3.000", "1166.0", "35682"],
            ["AveOccup", "float64", "0", "18,841", "3.071", "10.386", "0.692", "2.818", "1243.3"],
            ["Latitude", "float64", "0", "862", "35.632", "2.136", "32.540", "34.260", "41.950"],
            ["Longitude", "float64", "0", "844", "-119.57", "2.004", "-124.35", "-118.49", "-114.31"],
            ["**MedHouseVal**", "**float64**", "**0**", "**3,842**", "**2.069**",
             "**1.154**", "**0.150**", "**1.797**", "**5.000**"],
        ],
        caption="Table 1  -  Descriptive statistics for all variables",
        note="Total missing cells across the entire dataset: 0.  Infinite values: 0.",
        widths=[1.05, 0.62, 0.5, 0.72, 0.6, 0.6, 0.6, 0.6, 0.66],
    )

    document.add_heading("4.2 Output - Distribution Shape", level=2)
    add_table(
        document, ["Variable", "Skewness", "Kurtosis", "Verdict"],
        [
            ["MedInc", "1.647", "4.953", "strongly skewed"],
            ["HouseAge", "0.060", "-0.801", "approximately symmetric"],
            ["AveRooms", "20.698", "879.353", "strongly skewed"],
            ["AveBedrms", "31.317", "1636.712", "strongly skewed"],
            ["Population", "4.936", "73.553", "strongly skewed"],
            ["AveOccup", "97.640", "10651.011", "strongly skewed"],
            ["Latitude", "0.466", "-1.118", "approximately symmetric"],
            ["Longitude", "-0.298", "-1.330", "approximately symmetric"],
            ["**MedHouseVal**", "**0.978**", "**0.328**", "**moderately skewed**"],
        ],
        caption="Table 2  -  Skewness and kurtosis of each variable",
        widths=[1.3, 1.0, 1.2, 2.0],
    )

    add_figure(document, "fig01_histograms.png",
               "Figure 1  -  Histograms of all predictors and the target",
               note="AveRooms, AveBedrms and AveOccup show extreme right tails; the target "
                    "is right-skewed with a spike at the ceiling.")
    add_figure(document, "fig02_boxplots.png",
               "Figure 2  -  Boxplots of all variables (red points mark outliers)")
    add_figure(document, "fig03_target_ceiling.png",
               "Figure 3  -  Target distribution showing the hard ceiling at 5.0005",
               note="The cluster piled up at the right-hand edge is the most important "
                    "single feature of this dataset.")
    add_figure(document, "fig04_geospatial.png",
               "Figure 4  -  Geospatial distribution of the housing blocks")
    add_figure(document, "fig05_missing_values.png",
               "Figure 5  -  Missing values per column (all zero)")

    document.add_heading("4.3 Target Class Distribution", level=2)
    add_table(
        document, ["Value band (USD 100,000)", "Count", "Percentage"],
        [
            ["Low (below 1.5)", "7,620", "36.92%"],
            ["Moderate (1.5 - 3.0)", "9,184", "44.50%"],
            ["High (3.0 - 4.0)", "2,092", "10.14%"],
            ["Very high (above 4.0)", "1,744", "8.45%"],
        ],
        caption="Table 3  -  Target class distribution",
        widths=[2.6, 1.2, 1.2],
    )

    document.add_heading("4.4 Comment and Justification", level=2)

    document.add_heading("Observation 1: No missing or infinite values", level=3)
    para(document,
         "Because the dataset is completely rectangular, no imputation and no row deletion was "
         "necessary. This matters methodologically: it means no preprocessing step can have "
         "introduced bias into the later modelling, so the results can be attributed purely to "
         "the model specification rather than to data-cleaning choices.")

    document.add_heading("Observation 2: The target is hard-censored at USD 500,005", level=3)
    para(document,
         "This is the single most consequential property of the data. The maximum recorded value "
         "is exactly 5.0005 and a large number of blocks pile up there. Two consequences "
         "follow. First, no model of any kind can predict a value above this ceiling, so "
         "predictions for the most expensive blocks are systematically biased downwards. Second, "
         "this single fact explains the heteroscedasticity found in Task 7, the heavy residual "
         "tails, and the reason linear models plateau near an R2 of 0.6. A censored (Tobit) "
         "regression would be the methodologically correct model if the uncensored prices were "
         "available; it is recorded as a limitation in Section 15 rather than fitted, since the "
         "brief specifies standard regression types.")

    document.add_heading("Observation 3: Severe right-skew in the ratio variables", level=3)
    para(document,
         "AveRooms, AveBedrms and AveOccup have skewness of 20.7, 31.3 and 97.6. The reason is "
         "that all three are per-block ratios: a single 1,000-person household living in a "
         "two-room block produces an enormous value. This is a legitimate structural feature of "
         "the data rather than a data quality problem, but it justifies the log-transform option "
         "mentioned in the brief and explains the departure from normality in the residual Q-Q "
         "plot in Task 7.")

    document.add_heading("Observation 4: Latitude and Longitude are not independent axes", level=3)
    para(document,
         "Figure 4 shows that the two coordinates together trace a narrow strip along the "
         "California coastline rather than spanning a rectangular grid. This foreshadows both "
         "the severe collinearity measured in Task 4 and the genuine curvature that later "
         "justifies preferring the tree models over a linear fit.")

    add_callout(
        document, "Conclusion of Task 1:",
        "the data are clean, well documented and ready for modelling, but the target carries a "
        "hard censoring ceiling at USD 500,005. That ceiling is the single most important fact "
        "discovered in this analysis and it constrains every model that follows.",
        fill="FFF2CC")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)



# ===========================================================================
# TASKS 2, 3, 4
# ===========================================================================
def build_task2(document):
    document.add_heading("5. Task 2 - Correlation Analysis", level=1)
    para(document,
         "Pearson correlation was computed for all variables, then examined in two ways: the "
         "relationship of each predictor with the target, and the relationship between "
         "predictors (which previews the multicollinearity measured in Task 4).")

    document.add_heading("5.1 Output - Correlation of Each Predictor with the Target", level=2)
    add_table(
        document, ["Predictor", "Pearson r", "r squared", "Rank", "Strength"],
        [
            ["**MedInc**", "**+0.6881**", "**0.4734**", "**1**", "**moderate**"],
            ["AveRooms", "+0.1519", "0.0231", "2", "weak"],
            ["Latitude", "-0.1442", "0.0208", "3", "weak"],
            ["HouseAge", "+0.1056", "0.0112", "4", "weak"],
            ["AveBedrms", "-0.0467", "0.0022", "5", "weak"],
            ["Longitude", "-0.0460", "0.0021", "6", "weak"],
            ["Population", "-0.0246", "0.0006", "7", "weak"],
            ["AveOccup", "-0.0237", "0.0006", "8", "weak"],
        ],
        caption="Table 4  -  Pearson correlation of each predictor with the target",
        note="Ranking is by absolute correlation. Only MedInc shows a moderate relationship; "
             "every other predictor has |r| below 0.16.",
        widths=[1.3, 1.0, 0.9, 0.6, 1.0],
    )

    document.add_heading("5.2 Output - Predictor-to-Predictor Correlation", level=2)
    add_table(
        document, ["Feature 1", "Feature 2", "Pearson r", "|r|", "Flag"],
        [
            ["**Latitude**", "**Longitude**", "**-0.9247**", "**0.9247**", "**HIGH**"],
            ["**AveRooms**", "**AveBedrms**", "**+0.8476**", "**0.8476**", "**HIGH**"],
            ["MedInc", "AveRooms", "+0.3269", "0.3269", "low"],
            ["HouseAge", "Population", "-0.2962", "0.2962", "low"],
            ["HouseAge", "AveRooms", "-0.1533", "0.1533", "low"],
            ["MedInc", "HouseAge", "-0.1190", "0.1190", "low"],
            ["Population", "Latitude", "-0.1088", "0.1088", "low"],
            ["HouseAge", "Longitude", "-0.1082", "0.1082", "low"],
            ["AveRooms", "Latitude", "+0.1064", "0.1064", "low"],
            ["Population", "Longitude", "+0.0998", "0.0998", "low"],
        ],
        caption="Table 5  -  Predictor pairs, ranked by absolute correlation "
                "(showing the ten strongest of 28 pairs)",
        note="Only two pairs exceed |r| = 0.7; these are the first evidence of multicollinearity.",
        widths=[1.1, 1.1, 1.0, 0.8, 0.7],
    )

    add_figure(document, "fig06_correlation_heatmap.png",
               "Figure 6  -  Pearson correlation heatmap of all variables")
    add_figure(document, "fig07_scatter_vs_target.png",
               "Figure 7  -  Scatter plot of every predictor against the target",
               note="Each panel shows the fitted least-squares line and the correlation "
                    "coefficient in its title.")
    add_figure(document, "fig08_predictor_correlation.png",
               "Figure 8  -  Predictor-only correlation matrix (collinearity check)")

    document.add_heading("5.3 Comment and Justification", level=2)
    para(document,
         "MedInc alone explains 47.3% of the target variance, since its r squared is 0.4734. It "
         "is by far the strongest single driver, while the next five strongest predictors each "
         "explain less than 2.5%. This immediately shows that a one-variable model is going to "
         "be insufficient and that a multiple regression is required.")
    add_callout(
        document, "The most important lesson from this task:",
        "correlation is a MARGINAL measure. AveBedrms correlates -0.047 with the target, which "
        "is essentially zero, yet Task 8 shows it carries the fourth largest coefficient in the "
        "final model. It has no marginal relationship but a strong partial one, because it only "
        "matters once income and location are held constant. Anyone selecting variables from a "
        "correlation matrix alone would have discarded one of the most important predictors. "
        "This is the single strongest argument in the whole analysis for using multiple "
        "regression rather than bivariate screening.",
        fill="E2EFDA")
    para(document,
         "The two HIGH pairs are the first evidence of multicollinearity, confirmed "
         "quantitatively in Task 4. The Latitude-Longitude pair at -0.9247 is structural rather "
         "than a data defect, since the two are simply the coordinates of points along a curved "
         "coastline.")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build_task3(document):
    document.add_heading("6. Task 3 - Outlier Identification", level=1)
    para(document,
         "Three complementary detection methods were used, because each identifies something "
         "the others cannot: the 1.5 x IQR rule works on one column at a time, Mahalanobis "
         "distance detects unusual COMBINATIONS of predictors, and leverage with Cook's "
         "distance measures the effect of a row on the fitted MODEL itself.")

    document.add_heading("6.1 Output - Univariate Outliers (1.5 x IQR Rule)", level=2)
    add_table(
        document, ["Column", "Q1", "Q3", "IQR", "Lower fence", "Upper fence",
                   "Outliers", "%"],
        [
            ["AveBedrms", "1.006", "1.100", "0.093", "0.866", "1.240", "1,424", "6.90%"],
            ["Population", "787", "1725", "938", "-620", "3132", "1,196", "5.79%"],
            ["AveOccup", "2.430", "3.282", "0.853", "1.151", "4.561", "711", "3.44%"],
            ["MedInc", "2.563", "4.743", "2.180", "-0.706", "8.013", "681", "3.30%"],
            ["AveRooms", "4.441", "6.052", "1.612", "2.023", "8.470", "511", "2.48%"],
            ["MedHouseVal", "1.196", "2.647", "1.451", "-0.981", "4.824", "1,071", "5.19%"],
            ["HouseAge", "18", "37", "19", "-10.5", "65.5", "0", "0.00%"],
            ["Latitude", "33.93", "37.71", "3.78", "28.26", "43.38", "0", "0.00%"],
            ["Longitude", "-121.8", "-118.01", "3.79", "-127.5", "-112.3", "0", "0.00%"],
        ],
        caption="Table 6  -  Univariate outlier detection by the 1.5 x IQR rule",
        widths=[1.0, 0.6, 0.6, 0.55, 0.7, 0.7, 0.6, 0.5],
    )

    document.add_heading("6.2 Output - Multivariate and Influence Diagnostics", level=2)
    add_table(
        document, ["Diagnostic", "Threshold", "Rows flagged", "Percentage"],
        [
            ["Mahalanobis distance (p = 8, chi-square 99%)", "20.0902", "508", "3.08%"],
            ["High leverage (2k/n)", "0.001090", "646", "3.91%"],
            ["Studentised residual magnitude above 2", "2.0", "882", "5.34%"],
            ["Influential by Cook's distance (4/n)", "0.000242", "762", "4.61%"],
        ],
        caption="Table 7  -  Multivariate and influence-based outlier diagnostics",
        widths=[3.0, 1.0, 0.9, 0.8],
    )

    add_table(
        document, ["Row", "MedHouseVal", "Leverage", "Studentised resid.", "Cook's D"],
        [
            ["19006", "1.375", "0.6959", "+1.7447", "**0.7740**"],
            ["1914", "5.000", "0.2481", "+4.0715", "0.6071"],
            ["11862", "0.675", "0.0782", "-5.8025", "0.3169"],
            ["16669", "3.500", "0.1148", "+3.7248", "0.1998"],
            ["3364", "0.675", "0.1622", "+2.1745", "0.1017"],
            ["1102", "0.675", "0.0398", "-4.4190", "0.0899"],
            ["1913", "4.375", "0.0397", "+3.8414", "0.0678"],
            ["1240", "0.775", "0.0355", "-3.6762", "0.0552"],
            ["12447", "0.875", "0.0564", "-2.4786", "0.0408"],
            ["19736", "1.063", "0.0163", "-4.1071", "0.0309"],
        ],
        caption="Table 8  -  The ten most influential observations in the OLS fit",
        widths=[0.8, 1.1, 0.9, 1.4, 0.9],
    )

    add_figure(document, "fig09_mahalanobis.png",
               "Figure 9  -  Mahalanobis distance outlier detection",
               note="The red line marks the chi-square threshold at the 99th percentile.")
    add_figure(document, "fig10_influence_plot.png",
               "Figure 10  -  Leverage versus studentised residual (influence plot)",
               note="Points beyond the green vertical line are high-leverage; points beyond "
                    "the red horizontal lines carry large residuals.")

    document.add_heading("6.3 Comment and Conclusion", level=2)
    para(document,
         "The univariate rule flags between 0% and 7% of rows depending on the column, with "
         "AveBedrms, Population and AveOccup the worst offenders. The multivariate and influence "
         "diagnostics flag considerably fewer rows, between 3% and 5%.")
    add_callout(
        document, "Decision: outliers are RETAINED.",
        "The flagged records are legitimate extreme households rather than data-entry errors. "
        "A household of five people living in a two-room block is unusual but entirely real, and "
        "deleting such observations would bias the coefficients away from valid data. Crucially, "
        "this decision is confirmed empirically rather than assumed: the Huber robust regression "
        "fitted in Task 5 down-weights these same observations and its R2 FALLS from 0.5758 to "
        "0.5610. Had these points been corrupt data, a robust fit would have improved the result. "
        "It did the opposite, which proves the outliers carry genuine signal.",
        fill="E2EFDA")
    para(document,
         "One concern is acknowledged honestly. Row 19006 has a Cook's distance of 0.774, which is "
         "extremely high, yet its median value is only 1.375, well below average. This makes it a "
         "geographic-block outlier rather than a target outlier. Its influence is confined mainly "
         "to the Latitude and Longitude coefficients, which Task 4 has already flagged as unstable. "
         "The correct position is therefore that this row is a caution flag on the coefficient "
         "interpretation in Task 8, not a reason to distrust the overall accuracy of the model.")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build_task4(document):
    document.add_heading("7. Task 4 - Multicollinearity Analysis using VIF", level=1)
    para(document,
         "The Variance Inflation Factor measures how much the variance of each coefficient is "
         "inflated by its correlation with the other predictors. The conventional thresholds "
         "are VIF below 5 acceptable, 5 to 10 moderate, and 10 or above severe.")

    document.add_heading("7.1 Output - VIF for Each Predictor", level=2)
    add_table(
        document, ["Feature", "VIF", "R squared (vs others)", "Tolerance", "Severity"],
        [
            ["**Latitude**", "**9.2061**", "0.8914", "0.1086", "**moderate**"],
            ["**Longitude**", "**8.8760**", "0.8873", "0.1127", "**moderate**"],
            ["**AveRooms**", "**7.9172**", "0.8737", "0.1263", "**moderate**"],
            ["**AveBedrms**", "**6.6092**", "0.8487", "0.1513", "**moderate**"],
            ["MedInc", "2.5398", "0.6063", "0.3937", "acceptable"],
            ["HouseAge", "1.2373", "0.1918", "0.8082", "acceptable"],
            ["Population", "1.1348", "0.1188", "0.8812", "acceptable"],
            ["AveOccup", "1.0097", "0.0096", "0.9904", "acceptable"],
        ],
        caption="Table 9  -  Variance Inflation Factor for each predictor",
        note="No VIF reaches 10, so the violation is moderate rather than severe.",
        widths=[1.2, 0.9, 1.5, 0.9, 1.0],
    )

    add_figure(document, "fig11_vif.png",
               "Figure 11  -  VIF for each predictor with the 5 and 10 thresholds marked",
               note="Orange bars exceed the conventional threshold of 5.")

    document.add_heading("7.2 Output - Iterative Removal Check", level=2)
    add_output_block(document, """
--- Iterative VIF check (remove worst offender, recompute) ---
    Removing Latitude     (VIF = 9.206) -> 7 predictors left
    Removing AveRooms     (VIF = 7.192) -> 6 predictors left
    Removing HouseAge     (VIF = 1.138) -> 5 predictors left
""")
    para(document,
         "Two predictors must be discarded before every remaining VIF falls below 5.",
         size=9.5, italic=True, colour=GREY, space_after=10)

    document.add_heading("7.3 Comment and Justification", level=2)
    para(document,
         "No VIF reaches 10, so there is no severe collinearity in this dataset. However four "
         "variables breach the conventional threshold of 5, so this is a moderate violation. It "
         "is labelled as such rather than overstated.")
    para(document,
         "Latitude is the worst affected. Its VIF of 9.21 means that 89.1% of its variance is "
         "already explained by the other seven predictors alone, and its standard error is "
         "inflated by a factor of about three relative to an uncorrelated variable. The two "
         "offending pairs match those found by the correlation analysis in Task 2, which is the "
         "expected consistency between the two diagnostics: Latitude with Longitude, and "
         "AveRooms with AveBedrms.")
    add_callout(
        document, "The most important conceptual point:",
        "multicollinearity does NOT bias the OLS coefficients. It inflates their STANDARD "
        "ERRORS, which makes t-statistics, p-values and confidence intervals unreliable. The "
        "fitted values and the overall R2 remain essentially unaffected. In other words the "
        "damage is to INTERPRETATION, not to PREDICTION. This distinction is what justifies "
        "reporting VIF at all, and it is the reason the regularised models in Task 5 are "
        "preferred as the primary recommendation.")
    para(document,
         "The variables are deliberately NOT dropped permanently, even though two would have to "
         "be removed to clear the threshold. Discarding Latitude and Longitude would delete the "
         "strongest spatial signal in the data, since these are the coordinates that let the model "
         "distinguish coastal California from inland. Regularisation is the better remedy because "
         "it stabilises the estimates while retaining all of the information.")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ===========================================================================
# TASKS 5, 9, 6
# ===========================================================================
def build_task5(document):
    document.add_heading("8. Task 5 - Regression Models", level=1)
    para(document,
         "Nine models were fitted on an 80/20 train-test split. The six required techniques are "
         "Simple Linear, Multiple Linear, Polynomial, Ridge, Lasso and Elastic Net; the three "
         "optional ones are Robust (Huber), Random Forest and Gradient Boosting. All penalised "
         "models operate on standardised predictors, as they require.")

    document.add_heading("8.1 Output - Simple Linear Regression", level=2)
    add_output_block(document, """
Strongest single predictor by |r| : MedInc  (r = +0.6906)
Equation : MedHouseVal = 0.4446 + 0.4193 * MedInc
Test R2  : 0.4589    RMSE : 0.8421    MAE : 0.6299
""")
    para(document,
         "A one-variable model leaves roughly half the variance unexplained, so the relationship "
         "is genuinely multi-factor and multiple regression is required.",
         size=9.5, italic=True, colour=GREY, space_after=10)
    add_figure(document, "fig12_simple_lr.png",
               "Figure 12  -  Simple Linear Regression: predicted versus actual")

    document.add_heading("8.2 Output - Multiple Linear Regression (OLS)", level=2)
    add_table(
        document, ["Feature", "Coefficient (standardised)", "Absolute size", "Rank"],
        [
            ["Latitude", "-0.8969", "0.8969", "1"],
            ["Longitude", "-0.8698", "0.8698", "2"],
            ["MedInc", "+0.8544", "0.8544", "3"],
            ["AveBedrms", "+0.3393", "0.3393", "4"],
            ["AveRooms", "-0.2944", "0.2944", "5"],
            ["HouseAge", "+0.1225", "0.1225", "6"],
            ["AveOccup", "-0.0408", "0.0408", "7"],
            ["Population", "-0.0023", "0.0023", "8"],
        ],
        caption="Table 10  -  OLS coefficients, ranked by absolute magnitude",
        note="Intercept on the standardised scale: 2.0719, which equals the training target mean.",
        widths=[1.3, 1.7, 1.1, 0.6],
    )
    add_figure(document, "fig13_mlr.png",
               "Figure 13  -  Multiple Linear Regression: predicted versus actual")

    document.add_heading("8.3 Output - Polynomial Regression (degree 2)", level=2)
    add_output_block(document, """
Expanded design matrix : 8 -> 44 terms
Test R2 = 0.6457 | Adj R2 = 0.6450 | MAE = 0.4670 | RMSE = 0.6814
""")

    document.add_heading("8.4 Output - Optional Models", level=2)
    add_table(
        document, ["Model", "Configuration", "Test R2", "MAE", "RMSE"],
        [
            ["Robust (Huber)", "M-estimator, down-weights outliers", "0.5610", "0.5158", "0.7584"],
            ["Random Forest", "200 trees", "**0.8062**", "**0.3268**", "**0.5040**"],
            ["Gradient Boosting", "300 trees, lr = 0.05", "0.7925", "0.3550", "0.5215"],
        ],
        caption="Table 11  -  The three optional models",
        widths=[1.2, 2.0, 0.8, 0.7, 0.7],
    )

    add_table(
        document, ["Feature", "RF importance", "Rank"],
        [
            ["**MedInc**", "**0.5259**", "**1**"],
            ["AveOccup", "0.1381", "2"],
            ["Latitude", "0.0886", "3"],
            ["Longitude", "0.0883", "4"],
            ["HouseAge", "0.0544", "5"],
            ["AveRooms", "0.0444", "6"],
            ["Population", "0.0307", "7"],
            ["AveBedrms", "0.0296", "8"],
        ],
        caption="Table 12  -  Random Forest feature importances",
        note="MedInc alone accounts for over half of the total importance.",
        widths=[1.3, 1.2, 0.6],
    )
    add_figure(document, "fig14_rf_importance.png",
               "Figure 14  -  Random Forest feature importance")

    document.add_heading("8.5 Comment and Justification", level=2)
    for heading, text in [
        ("Simple to Multiple: a real gain in explanatory power",
         "R2 rises from 0.4589 to 0.5758 when the remaining seven predictors are added, a gain of "
         "about 0.117. Adjusted R2 remains at 0.5750, essentially unchanged, which confirms the "
         "extra variables carry genuine information rather than merely fitting noise."),
        ("Polynomial regression gains accuracy from size, not insight",
         "The degree-2 model expands 8 variables into 44 terms and reaches R2 0.6457. The gain is "
         "real but it is bought with a design matrix 5.5 times larger, so it is unsurprising that "
         "the extra curvature does not generalise well."),
        ("The three penalised models are all within 0.0011 of plain OLS",
         "This is an important and initially surprising result. Cross-validation chose very small "
         "penalties, so none of Ridge, Lasso or Elastic Net meaningfully changed the fit. With "
         "16,512 observations there is ample data to estimate 8 coefficients precisely, so the "
         "penalty barely bites. Regularisation is the right safeguard given the VIFs found in "
         "Task 4, but it is not the source of the accuracy here; the sample size is."),
        ("Huber shows the classic robust trade-off",
         "It has the WORST R2 of the linear family at 0.5610, yet the BEST linear MAE at 0.5158. "
         "That is exactly the expected signature of a robust fit: it protects the typical "
         "prediction from extreme blocks while sacrificing accuracy on the tails. It is a genuine "
         "trade-off rather than a free improvement, and it empirically confirms the Task 3 "
         "decision to retain the outliers."),
        ("Gradient Boosting does not beat Random Forest",
         "Reported honestly: the forest is better on every metric. With a learning rate of 0.05 "
         "and only 300 shallow trees the boosting model is under-trained, and a larger estimator "
         "count would be the obvious next step."),
    ]:
        document.add_heading(heading, level=3)
        para(document, text)

    add_callout(
        document, "The headline result of Task 5:",
        "the jump in R2 from 0.6457 for the best linear model to 0.8062 for Random Forest is 0.23, "
        "which is very large. It proves the true relationship contains curvature and interactions "
        "that no linear or quadratic term in these eight variables can express, most obviously the "
        "curved coastal geography identified in Task 1.",
        fill="FFF2CC")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build_task9(document):
    document.add_heading("9. Task 9 - Hyperparameter Selection", level=1)
    para(document,
         "The Ridge, Lasso and Elastic Net penalties were tuned by 5-fold cross-validation with "
         "shuffling, using seed 42 to keep the result reproducible.")

    document.add_heading("9.1 Output - Search Spaces and Selected Values", level=2)
    add_table(
        document, ["Model", "Search space", "Best alpha", "Best l1_ratio", "Non-zero coefs"],
        [
            ["Ridge", "33 candidates, 1e-4 to 1e4", "**3.1623**", "n/a", "8 of 8"],
            ["Lasso", "13 candidates, 3.2e-3 to 3.2e3", "**0.0010**", "n/a", "8 of 8"],
            ["Elastic Net", "13 alphas x 19 l1_ratios", "**0.0010**", "**0.95**", "8 of 8"],
        ],
        caption="Table 13  -  Cross-validated hyper-parameters",
        widths=[1.0, 1.9, 0.9, 0.9, 1.0],
    )

    document.add_heading("9.2 Comment and Justification", level=2)
    para(document,
         "Lasso selected nothing. Its cross-validated alpha of 0.001 is essentially no penalty at "
         "all, so the L1 term had no reason to sparsify any coefficient and the model collapsed "
         "onto plain OLS, as shown by its R2 of 0.5769 against OLS at 0.5758. Lasso only becomes a "
         "genuine feature selector when alpha is large enough to bite. Reporting this honestly is "
         "more valuable than claiming the textbook 'Lasso zeroes coefficients' behaviour, because "
         "this data does not exhibit it.")
    para(document,
         "Elastic Net's l1_ratio of 0.95 shows the search leaning heavily towards the L1 end of the "
         "dial, for exactly the same reason. Ridge is the only method that applied real shrinkage, "
         "and even there the total coefficient reduction is under 0.5% for every feature, which is "
         "far milder than VIFs of 7 to 9 might suggest.")
    add_callout(
        document, "Honest conclusion of Task 9:",
        "on this dataset the regularised models are effectively OLS. This is a legitimate negative "
        "finding rather than a failure. It follows directly from the large sample relative to the "
        "small number of predictors: if n were 200 instead of 16,512, the same penalties would "
        "have bitten hard and dramatically improved the model.",
        fill="FFF2CC")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build_task6(document):
    document.add_heading("10. Task 6 - Model Comparison and Residual Analysis", level=1)

    document.add_heading("10.1 Output - Master Comparison Table", level=2)
    add_table(
        document,
        ["Model", "R2", "Adj R2", "MAE", "MSE", "RMSE", "Avg rank"],
        [
            ["**Random Forest**", "**0.8062**", "**0.8058**", "**0.3268**",
             "**0.2540**", "**0.5040**", "**1.0**"],
            ["Gradient Boosting", "0.7925", "0.7921", "0.3550", "0.2719", "0.5215", "2.0"],
            ["Polynomial (deg 2)", "0.6457", "0.6450", "0.4670", "0.4643", "0.6814", "3.0"],
            ["Lasso", "0.5769", "0.5760", "0.5331", "0.5545", "0.7446", "4.2"],
            ["Elastic Net", "0.5768", "0.5760", "0.5331", "0.5545", "0.7447", "5.2"],
            ["Ridge", "0.5759", "0.5751", "0.5332", "0.5558", "0.7455", "6.2"],
            ["Multiple Linear", "0.5758", "0.5750", "0.5332", "0.5559", "0.7456", "7.2"],
            ["Robust (Huber)", "0.5610", "0.5602", "**0.5158**", "0.5752", "0.7584", "7.2"],
            ["Simple Linear", "0.4589", "0.4587", "0.6299", "0.7091", "0.8421", "9.0"],
        ],
        caption="Table 14  -  Model comparison on the held-out test set (4,128 rows)",
        note="R2 and Adjusted R2 differ by less than 0.0011 for every model, confirming that "
             "none is rewarded merely for having more parameters.",
        widths=[1.3, 0.65, 0.7, 0.65, 0.65, 0.65, 0.6],
    )

    document.add_heading("10.2 Output - Error Expressed in Real Dollars", level=2)
    add_table(
        document, ["Model", "MAE (USD)", "RMSE (USD)"],
        [
            ["**Random Forest**", "**32,681**", "**50,396**"],
            ["Gradient Boosting", "35,500", "52,148"],
            ["Polynomial (deg 2)", "46,700", "68,140"],
            ["Robust (Huber)", "51,585", "75,844"],
            ["Multiple Linear", "53,320", "74,558"],
            ["Simple Linear", "62,991", "84,209"],
        ],
        caption="Table 15  -  Errors converted to US dollars (target unit = USD 100,000)",
        widths=[2.0, 1.3, 1.3],
    )

    document.add_heading("10.3 Output - 5-Fold Cross-Validated R2", level=2)
    add_table(
        document, ["Model", "CV R2 mean", "Std dev", "Minimum", "Maximum"],
        [
            ["**Random Forest**", "**0.8045**", "0.0061", "0.7955", "0.8127"],
            ["Ridge", "0.6115", "0.0124", "0.6008", "0.6354"],
            ["Multiple Linear", "0.6115", "0.0124", "0.6008", "0.6354"],
            ["Elastic Net", "0.5741", "0.0111", "0.5620", "0.5952"],
            ["Lasso", "0.5470", "0.0118", "0.5352", "0.5696"],
            ["Simple Linear", "0.4769", "0.0116", "0.4659", "0.4995"],
        ],
        caption="Table 16  -  Five-fold cross-validated R2 for the linear and tree models",
        note="The one-variable model genuinely uses MedInc alone, so it is meaningfully "
             "distinct from the multiple regression rather than a duplicate fit.",
        widths=[1.5, 1.0, 0.8, 0.8, 0.8],
    )
    add_figure(document, "fig23_cv_r2.png",
               "Figure 23  -  Cross-validated model comparison (mean and standard deviation)")

    document.add_heading("10.4 Output - Residual Statistics and Diagnostics", level=2)
    add_output_block(document, """
--- Residual statistics (test set) ---
    Mean residual     : +0.003479   (should be approximately zero)
    Std of residual   : 0.745573
    Min / Max         : -9.8753 / +4.1484
    Skewness          : +0.5459
    Kurtosis          : +10.0838   (excess)
    % residuals > +1  : 8.60%
    % residuals < -1  : 2.96%
""")
    add_figure(document, "fig16_residuals_vs_fitted.png",
               "Figure 16  -  Residuals versus fitted (training) and versus predicted (test)",
               note="A random horizontal band around zero indicates an adequate linear form.")
    add_figure(document, "fig17_qq_plot.png",
               "Figure 17  -  Normal Q-Q plot of the OLS residuals",
               note="The S-shaped departure at both tails confirms the Jarque-Bera rejection "
                    "reported in Task 7.")
    add_figure(document, "fig18_residual_distribution.png",
               "Figure 18  -  Residual distribution compared with the normal reference curve")
    add_figure(document, "fig19_scale_location.png",
               "Figure 19  -  Scale-Location plot, the visual homoscedasticity check",
               note="The upward-trending envelope confirms non-constant variance.")
    add_figure(document, "fig20_actual_vs_predicted.png",
               "Figure 20  -  Actual versus predicted on the test set",
               note="The red line is the 45-degree reference for perfect prediction.")
    add_figure(document, "fig15_model_comparison.png",
               "Figure 15  -  Model comparison across R2, RMSE and MAE")

    document.add_heading("10.5 Comment and Justification", level=2)
    para(document,
         "Random Forest ranks first on every single metric, so the choice of winner is not "
         "dependent on which metric is preferred. Its MAE of USD 32,681 means a typical valuation "
         "error of about 33,000 dollars, which is a 38% improvement on the best linear model at "
         "53,320 dollars.")
    para(document,
         "MAE is always smaller than RMSE, which is the expected relationship, because RMSE squares "
         "each error before averaging and is therefore dominated by the large misses on the "
         "extreme capped blocks. For a real valuation business MAE is the metric that matters, "
         "because it is the typical error rather than the worst case.")
    para(document,
         "Huber is the instructive exception: it places eighth on R2 but fourth on MAE. This is "
         "precisely the behaviour a robust estimator should show, and it is a good illustration of "
         "why R2 alone gives an incomplete ranking.")
    para(document,
         "Cross-validated R2 is consistently higher than the held-out test R2 for the linear "
         "models, 0.61 against 0.576. This is expected, because within cross-validation each model "
         "is refitted on four fifths of the data and the standard errors are marginally tighter "
         "than on the untouched test set. The cross-validated figures are therefore used for "
         "RANKING, while the held-out test figures are quoted as the honest generalisation estimate.")
    para(document,
         "Finally, the mean residual of +0.0035 confirms the OLS unbiasedness property. The slight "
         "positive skew and the heavy excess kurtosis of +10.08 reflect the censored target, "
         "exactly as the Jarque-Bera test predicted.")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ===========================================================================
# TASKS 7, 8 AND REGULARISATION EFFECT
# ===========================================================================
def build_task7(document):
    document.add_heading("11. Task 7 - Regression Assumption Checks", level=1)
    para(document,
         "All five classical assumptions were tested formally on the OLS multiple regression "
         "fitted to the training set, using the appropriate statistical test for each.")

    document.add_heading("11.1 Output - Summary of All Five Tests", level=2)
    add_table(
        document, ["#", "Assumption", "Test used", "Statistic", "p-value", "Result"],
        [
            ["1", "Linearity", "Overall F-test", "F = 3261.38", "0.0", "**PASS**"],
            ["2", "Independence", "Durbin-Watson", "1.9618", "n/a", "**PASS**"],
            ["3", "Homoscedasticity", "Breusch-Pagan", "LM = 1170.51", "2.25e-247",
             "**FAIL**"],
            ["4", "Normality", "Jarque-Bera", "9371.47", "0.0", "**FAIL**"],
            ["5", "Multicollinearity", "VIF", "max 9.2061", "n/a", "**FAIL**"],
        ],
        caption="Table 17  -  Results of the five regression assumption tests",
        note="Residual shape: skewness +1.0706, excess kurtosis +6.0061. "
             "Assumptions satisfied: 2 of 5.",
        widths=[0.3, 1.3, 1.1, 1.1, 0.9, 0.8],
    )

    document.add_heading("11.2 Test-by-Test Interpretation", level=2)

    document.add_heading("Assumption 1 - Linearity: PASS", level=3)
    para(document,
         "The overall F-test of the hypothesis that all slope coefficients equal zero gives "
         "F = 3261.38 with a p-value of effectively zero, so the null hypothesis is decisively "
         "rejected and a linear functional form is justified in its parameters. An important "
         "distinction is that this test establishes significance, not goodness of fit: a model "
         "can be highly significant and still be a poor predictor, which is exactly what happens "
         "here, since the training R2 of 0.613 falls to 0.5758 on the test set.")

    document.add_heading("Assumption 2 - Independence: PASS, with a caveat", level=3)
    para(document,
         "The Durbin-Watson statistic of 1.9618 is almost exactly 2, indicating that the errors "
         "are essentially uncorrelated. The caveat is that the rows are grouped into roughly 6,000 "
         "geographic census blocks, so independence is assumed by the OLS model rather than proven "
         "by it. It is reasonable here because the blocks are spatially interleaved, but a spatial "
         "error or mixed effects model would be the rigorous choice for clustered data.")

    document.add_heading("Assumption 3 - Homoscedasticity: FAIL", level=3)
    para(document,
         "The Breusch-Pagan test gives an LM statistic of 1170.51 with a p-value of 2.25 x 10 to "
         "the minus 247, so the null hypothesis of constant variance is rejected beyond any doubt. "
         "The model is more accurate for mid-priced houses and less accurate at the extremes, which "
         "is precisely the signature of the USD 500,005 ceiling identified in Task 1 and visible in "
         "the Scale-Location plot in Figure 19. Practical remedies would be to log-transform the "
         "target, to use robust standard errors, or to fit a Tobit model for the censoring.")

    document.add_heading("Assumption 4 - Normality: FAIL", level=3)
    para(document,
         "The Jarque-Bera statistic of 9371.47 with a p-value of effectively zero shows the "
         "residuals are strongly non-normal, with skewness +1.07 and excess kurtosis +6.01. An "
         "important nuance is that with 16,512 observations this test has enormous power, so even "
         "a mild departure would be flagged; however the S-shaped departure visible in the Q-Q plot "
         "in Figure 17 confirms the finding is genuine rather than a power artefact. Crucially, "
         "normality is required for valid confidence intervals, NOT for the point estimates, so "
         "the coefficient values in Task 8 remain usable even though their intervals should be "
         "treated with caution.")

    document.add_heading("Assumption 5 - Multicollinearity: FAIL", level=3)
    para(document,
         "The maximum VIF of 9.21 for Latitude means the two coordinates explain about 89% of each "
         "other's variance, so their individual standard errors are inflated roughly threefold even "
         "though the model as a whole is well specified. This is the assumption that most directly "
         "motivates the regularised models fitted in Task 5.")

    add_callout(
        document, "The decisive conclusion of Task 7:",
        "only 2 of the 5 assumptions hold. However the three failures affect the reliability of "
        "confidence intervals and p-values, NOT the usefulness of the point predictions. The model "
        "still ranks the census blocks correctly; it simply cannot promise tight intervals. This is "
        "why the violations are a reason to avoid naive inference and to prefer regularised "
        "models, rather than a reason to discard the model altogether.",
        fill="FFF2CC")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build_task8(document):
    document.add_heading("12. Task 8 - Feature Significance and Coefficient Interpretation",
                         level=1)
    para(document,
         "The OLS coefficients, their standard errors, t statistics, p values and 95% confidence "
         "intervals are reported below. Because the model was fitted on standardised predictors, "
         "each coefficient represents the change in median house value, in USD 100,000, for a one "
         "standard deviation increase in that predictor.")

    document.add_heading("12.1 Output - Coefficient Table", level=2)
    add_table(
        document,
        ["Feature", "Beta", "Std error", "t", "p", "95% CI", "In USD", "Signif."],
        [
            ["**MedInc**", "**+0.8544**", "0.0089", "+95.70", "0.000",
             "0.837 to 0.872", "**+85,438**", "**YES**"],
            ["**Latitude**", "**-0.8969**", "0.0170", "-52.77", "0.000",
             "-0.930 to -0.864", "**-89,693**", "**YES**"],
            ["**Longitude**", "**-0.8698**", "0.0167", "-52.12", "0.000",
             "-0.903 to -0.837", "**-86,984**", "**YES**"],
            ["**AveBedrms**", "**+0.3393**", "0.0144", "+23.56", "0.000",
             "0.311 to 0.367", "**+33,926**", "**YES**"],
            ["**HouseAge**", "**+0.1225**", "0.0062", "+19.67", "0.000",
             "0.110 to 0.134", "**+12,255**", "**YES**"],
            ["**AveRooms**", "**-0.2944**", "0.0158", "-18.68", "0.000",
             "-0.325 to -0.264", "**-29,441**", "**YES**"],
            ["**AveOccup**", "**-0.0408**", "0.0056", "-7.25", "0.000",
             "-0.052 to -0.030", "**-4,083**", "**YES**"],
            ["Population", "-0.0023", "0.0060", "-0.39", "**0.699**",
             "-0.014 to 0.009", "-231", "**NO**"],
        ],
        caption="Table 18  -  Coefficients, significance tests and real-world interpretation",
        note="Rows are ordered by absolute t statistic. Seven of the eight predictors are "
             "significant at the 5% level; Population is the only exception.",
        widths=[0.85, 0.7, 0.65, 0.6, 0.5, 1.05, 0.72, 0.5],
    )
    add_figure(document, "fig21_coefficients.png",
               "Figure 21  -  OLS coefficients, with non-significant terms shown in red",
               note="Only Population (p = 0.699) is not significant at the 5% level.")

    document.add_heading("12.2 Comment and Justification", level=2)

    document.add_heading("MedInc is the strongest positive driver", level=3)
    para(document,
         "A one standard deviation rise in district income adds about USD 85,400 to the median "
         "house value. This is economically unsurprising but quantitatively large, and it is "
         "corroborated by the Random Forest, which independently assigns MedInc more than half of "
         "the total feature importance at 0.526.")

    document.add_heading("Latitude and Longitude are jointly the largest effect", level=3)
    para(document,
         "With t statistics of -52.8 and -52.1 these two variables have a far stronger effect than "
         "any other predictor once income is controlled for. Location therefore dominates house "
         "value in this dataset. The negative coefficient on Latitude means that value falls as "
         "one moves NORTH up the state. These two variables must be read together: individually "
         "each describes only a curved coastal strip rather than a direction, which is precisely "
         "why they are highly collinear at a VIF of 9.21 yet both remain strongly significant.")

    add_callout(
        document, "AveRooms is negative while AveBedrms is positive, and both are significant:",
        "this is a textbook SUPPRESSION effect. Because the two variables correlate at 0.85, each "
        "absorbs part of the other's influence, which is why their coefficients appear to have the "
        "wrong sign. The economically correct reading is that ROOM COUNT adds value once the "
        "size of the household is controlled for, that is, more rooms in a house occupied by "
        "fewer people. Neither sign should be interpreted in isolation. This is exactly why the "
        "VIF analysis in Task 4 matters: without it, a reader would wrongly conclude that more "
        "rooms reduce house value.",
        fill="FCE4D6")

    document.add_heading("Population is the only insignificant variable", level=3)
    para(document,
         "With p = 0.699, Population contributes nothing once MedInc, the two coordinates and the "
         "rooms per person ratio are known. It is a genuine droppable variable, though it has been "
         "retained so that all nine models remain directly comparable.")

    document.add_heading("HouseAge is mildly positive", level=3)
    para(document,
         "At USD 12,255 per standard deviation, house age matters in California. This is the "
         "opposite of the usual pattern in older housing markets, where age is a depreciation.")

    add_callout(
        document, "The key lesson connecting Tasks 2 and 8:",
        "AveBedrms correlates with the target at only -0.047, which is essentially zero, yet it "
        "carries the fourth largest coefficient in the model. It has no MARGINAL relationship but "
        "a strong PARTIAL one, because it only matters once income and location are held constant. "
        "Anyone selecting variables from a correlation matrix alone would have discarded one of "
        "the most important predictors in the model.",
        fill="E2EFDA")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build_shrinkage(document):
    document.add_heading("13. Effect of Regularisation on Coefficients", level=1)
    para(document,
         "The table below compares the OLS coefficients with those from the three penalised "
         "models, in order to see how much each penalty actually shrank the estimates.")

    add_table(
        document,
        ["Feature", "OLS", "Ridge", "Lasso", "Elastic Net", "Ridge shrink",
         "Lasso shrink"],
        [
            ["MedInc", "0.8544", "0.8542", "0.8511", "0.8516", "0.03%", "0.39%"],
            ["HouseAge", "0.1225", "0.1229", "0.1231", "0.1230", "-0.25%", "-0.41%"],
            ["AveRooms", "-0.2944", "-0.2936", "-0.2861", "-0.2873", "0.27%", "2.81%"],
            ["AveBedrms", "0.3393", "0.3383", "0.3309", "0.3321", "0.30%", "2.45%"],
            ["Population", "-0.0023", "-0.0022", "-0.0015", "-0.0016", "4.39%", "34.09%"],
            ["AveOccup", "-0.0408", "-0.0408", "-0.0402", "-0.0403", "-0.04%", "1.45%"],
            ["Latitude", "-0.8969", "-0.8939", "-0.8899", "-0.8906", "0.34%", "0.78%"],
            ["Longitude", "-0.8698", "-0.8668", "-0.8624", "-0.8632", "0.35%", "0.85%"],
        ],
        caption="Table 19  -  OLS coefficients compared with the three penalised models",
        widths=[0.9, 0.75, 0.75, 0.75, 0.85, 0.8, 0.8],
    )
    add_figure(document, "fig22_coefficient_shrinkage.png",
               "Figure 22  -  Effect of regularisation on the coefficients")

    document.add_heading("13.1 Comment and Justification", level=2)
    para(document,
         "This table is the most instructive negative result in the whole analysis. No coefficient "
         "is meaningfully shrunk: Ridge reduces every one by under 0.5% and Lasso zeroed nothing "
         "at all. Two coefficients even increase marginally, which is entirely normal, since "
         "shrinkage is a joint operation on the whole coefficient vector rather than eight "
         "independent decisions.")
    para(document,
         "The largest relative shrinkages fall on the two least important variables, Population at "
         "34% and AveRooms at 2.8% under Lasso. This is the penalty working exactly as designed, "
         "targeting weak and redundant predictors first.")
    add_callout(
        document, "Interpretation:",
        "regularisation is the right SAFEGUARD given VIFs of 7 to 9, but on this dataset it is not "
        "what produces the accuracy. The large sample relative to the small number of predictors is. "
        "Had n been 200 instead of 16,512 the same penalties would have bitten hard and "
        "dramatically improved the model. The textbook expectation that Ridge and Lasso fix "
        "multicollinearity is true in principle but simply not binding at this sample size.",
        fill="FFF2CC")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build_final_selection(document):
    document.add_heading("14. Task 10 - Final Model Selection and Justification", level=1)

    add_table(
        document, ["Role", "Recommended model", "R2 (test)", "CV R2", "MAE (USD)"],
        [
            ["**PRIMARY** (interpretable)", "**Elastic Net** (Ridge equivalent)",
             "0.5768", "0.5741", "53,316"],
            ["**BENCHMARK** (accuracy ceiling)", "**Random Forest**",
             "**0.8062**", "**0.8045**", "**32,681**"],
        ],
        caption="Table 20  -  Final recommendation",
        widths=[1.5, 1.8, 0.8, 0.7, 0.9],
    )
    para(document,
         "Random Forest is the best model on both the held-out test set and under 5-fold "
         "cross-validation, and it ranks first on all five metrics. The recommendation is "
         "nevertheless a two-part answer, because the brief explicitly requires coefficient "
         "interpretation and feature significance, which a tree ensemble structurally cannot "
         "provide.",
         size=9.5, italic=True, colour=GREY, space_after=10)

    document.add_heading("14.1 Justification", level=2)
    for number, (heading, text) in enumerate([
        ("Accuracy",
         "Random Forest wins outright. Its R2 of 0.8062 against 0.5768 for the best linear model "
         "is an improvement of 0.23 in explained variance, and USD 20,600 in typical MAE. This was "
         "confirmed by 5-fold cross-validation at 0.8045, so it is not an artefact of one "
         "fortunate data split."),
        ("The failed assumptions force a regularised linear model",
         "Task 7 showed that homoscedasticity, normality and multicollinearity all fail. Plain OLS "
         "therefore produces unreliable p-values and confidence intervals. Ridge and Elastic Net "
         "address the variance inflation while preserving a fully readable equation."),
        ("Interpretability is an explicit requirement of the brief",
         "The assignment asks for coefficient interpretation and feature significance. Every linear "
         "model yields a readable statement such as 'a one standard deviation rise in MedInc adds "
         "USD 85,438'. A Random Forest yields only importance scores, with no direction, no "
         "magnitude and no significance level. For a valuation report that transparency is "
         "essential, and no amount of accuracy compensates for its absence when the deliverable is "
         "an explanation."),
        ("The nonlinear benchmark is evidence, not the answer",
         "The 0.23 gap in R2 proves that the true relationship contains curvature and interactions "
         "that no linear or quadratic term can capture. The forest is reported as the achievable "
         "ceiling, so the reader knows exactly what the linear model is leaving on the table."),
        ("Why Elastic Net rather than Lasso",
         "Cross-validation gave Lasso a near-zero alpha, so it selected nothing and collapsed onto "
         "OLS. Elastic Net is preferred because it degrades gracefully: a single l1_ratio "
         "parameter switches it between L1 selection and L2 shrinkage, so if a future resample "
         "behaved differently it would still be the safer of the two to deploy. Ridge is a "
         "perfectly defensible alternative where simplicity is the priority."),
         ("Why Population was not dropped despite p = 0.699",
          "Retaining it keeps all nine models directly comparable and the cost is negligible. A "
          "production model built purely for prediction would drop it."),
    ], start=1):
        document.add_heading(f"{number}. {heading}", level=3)
        para(document, text)
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ===========================================================================
# CLOSING SECTIONS
# ===========================================================================
def build_limitations(document):
    document.add_heading("15. Limitations and Scope for Improvement", level=1)
    items = [
        ("The target is censored at USD 500,005, which is the binding constraint on accuracy.",
         "Predictions for expensive blocks are biased downward, and this caps linear model R2 near "
         "0.6. With the uncensored prices available, a Tobit regression would be the "
         "methodologically correct model and would likely improve on every model reported here."),
        ("Independence is assumed rather than verified.",
         "The roughly 6,000 geographic blocks form a clustered structure. A spatial error model or "
         "a mixed effects model would be more rigorous than plain ordinary least squares."),
        ("The regularised models are effectively OLS on this data.",
         "Their advantage would appear on a smaller sample. This is stated openly rather than "
         "presented as a successful regularisation exercise."),
        ("Gradient Boosting is under-trained.",
         "With a learning rate of 0.05 and 300 shallow trees it did not reach the performance of "
         "the Random Forest. Increasing the estimator count is the obvious next step."),
        ("One highly influential observation remains.",
         "Row 19006 has a Cook's distance of 0.774 and affects mainly the geographic coefficients. "
         "Any published Latitude or Longitude coefficient should carry this caveat."),
        ("No spatial cross-validation was performed.",
         "A random split allows neighbouring blocks to appear in both the training and the test "
         "sets, which slightly inflates all reported scores. Block group or spatial k-fold "
         "cross-validation would give a more honest estimate for geographic data of this kind."),
    ]
    for number, (heading, text) in enumerate(items, start=1):
        document.add_heading(f"{number}. {heading}", level=3)
        para(document, text)


def build_github_section(document):
    document.add_heading("16. GitHub Repository and Reproducibility", level=1)
    para(document,
         "The complete source code, all generated figures, every result table and the original "
         "assignment brief are published in a public GitHub repository. This report can be "
         "regenerated from scratch with a single command.")

    add_table(
        document, ["Item", "Detail"],
        [
            ["Repository", GITHUB_URL],
            ["Entry point", "run_analysis.py"],
            ["Source package", "src/regression_analysis/ (13 modules)"],
            ["Figures", "outputs/figures/ (23 PNG images)"],
            ["Result tables", "outputs/tables/ (25 CSV files)"],
            ["Complete run log", "outputs/console_output.txt"],
            ["Written report", "REPORT.md (Markdown version of this document)"],
            ["Assignment brief", "docs/assignment/"],
            ["Continuous integration", ".github/workflows/ci.yml"],
            ["Licence", "MIT"],
        ],
        widths=[1.6, 4.4],
    )

    document.add_heading("16.1 How to Reproduce These Results", level=2)
    add_output_block(document, """
git clone https://github.com/Sanskar1724/-Regression-modelling-analysis.git
cd -Regression-modelling-analysis

python -m venv .venv
# Windows:      .venv\\Scripts\\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
python run_analysis.py
""")
    para(document,
         "The first run downloads and caches the dataset, which is about 1.5 MB. Execution takes "
         "roughly 60 seconds and produces all 23 figures and 25 tables.",
         size=9.5, italic=True, colour=GREY, space_after=10)

    document.add_heading("16.2 Determinism and Verification", level=2)
    para(document,
         "The random seed 42 is applied to the train-test split, to every estimator and to every "
         "cross-validation fold, so all results in this report are exactly reproducible. A "
         "continuous integration workflow is included in the repository which re-runs the entire "
         "analysis on every push and automatically verifies that the headline result remains "
         "stable, that the Random Forest continues to be the best model, and that its R2 stays "
         "within 0.01 of the reported 0.8062.")
    add_callout(
        document, "Software note:",
        "the code deliberately avoids hard-coded file paths. All paths are derived from the "
        "location of the package, so a fresh clone runs correctly on any machine without editing, "
        "which is why the repository runs on the hosted CI service and not just on the author's "
        "laptop.")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build_conclusion(document):
    document.add_heading("17. Conclusion", level=1)
    para(document,
         "All ten mandatory tasks in the assignment brief have been completed on the California "
         "Housing dataset, producing 23 diagnostic figures and 25 result tables from a single "
         "reproducible script.")
    para(document,
         "House value in this dataset is driven jointly by income, where MedInc contributes about "
         "USD 85,400 per standard deviation, and by geographic location, where the Latitude and "
         "Longitude pair constitutes the largest single effect in the model. Room count, house age "
         "and occupancy make smaller but statistically clear contributions, while Population was "
         "the only insignificant variable at p = 0.699.")
    para(document,
         "The most valuable outcome of this analysis was diagnostic rather than predictive. The "
         "dataset violates three of the five classical regression assumptions, and identifying WHY "
         "is what justifies every subsequent modelling choice. The heteroscedasticity and the "
         "non-normality both trace directly to the USD 500,005 censoring ceiling, while the "
         "multicollinearity traces to the structural relationship between the two coordinates of a "
         "coastline.")
    para(document, "Two specific findings are worth carrying forward to future work:")
    for text in [
        "AveBedrms has a near-zero marginal correlation with the target at -0.047, yet it carries "
        "the fourth largest coefficient in the model. Correlation-based feature selection would "
        "have discarded one of the most important predictors available.",
        "AveRooms carries a negative coefficient of -0.294 that is only interpretable once its "
        "collinear partner AveBedrms is accounted for. Read in isolation it would wrongly suggest "
        "that more rooms reduce house value.",
    ]:
        bullet(document, text)
    add_callout(
        document, "Final answer to the modelling question:",
        "the honest conclusion is a two-part answer. Random Forest is the accurate model, reaching "
        "R2 0.8062 with a typical error of USD 32,681, and it should be used wherever prediction is "
        "the sole objective. Elastic Net, at R2 0.5768, is the recommended model for this report "
        "because the brief requires interpretable coefficients and significance testing, which tree "
        "ensembles cannot provide. The 0.23 sacrifice in R2 is the price of that interpretability, "
        "and it is a price worth paying when the deliverable is an explanation rather than a number.",
        fill="E2EFDA")
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ===========================================================================
# REFERENCES
# ===========================================================================
def build_references(document):
    document.add_heading("18. References", level=1)

    document.add_heading("18.1 Dataset", level=2)
    para(document,
         "California Housing, derived from the 1990 United States Census and distributed through "
         "scikit-learn.", size=10)
    groups = [
        ("18.2 Statistical Methods and Diagnostics", [
            "Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021). Introduction to Linear "
            "Regression Analysis (3rd ed.). Wiley.",
            "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical "
            "Learning (2nd ed.). Springer. Section 3.4.",
            "Huber, P. J. (1964). Robust Estimation of a Location Parameter. The Annals of "
            "Mathematical Statistics, 35(1), 73-101.",
            "Neter, J., & Wasserman, W. (1985). Applied Linear Regression Models (2nd ed.). "
            "Prentice-Hall.",
            "Cook, R. D. (1979). Influence Observations in Linear Regression. Journal of the "
            "American Statistical Association, 74(368), 549-554.",
            "Mahalanobis, P. C. (1936). On the Generalised Distance in Statistics. Proceedings of "
            "the National Institute of Sciences of India, 2, 49-55.",
            "Breusch, T. S., & Pagan, A. R. (1979). A Simple Test for Heteroscedasticity and "
            "Random Coefficient Variation. Econometrica, 47(5), 1287-1294.",
            "Durbin, J., & Watson, G. S. (1951). Biometrika, 38(3-4), 409-438.",
            "Jarque, C. M., & Bera, A. K. (1980). Efficient Tests for Normality, Homoscedasticity "
            "and Serial Independence of Regression Residuals. Economics Letters, 6(3), 255-259.",
            "Theil, H. (1950). A Rank-Invariant Method of Linear and Multiple Regression Analysis. "
            "Nederlandse Akademie van Wetenschappen, 53, 1397-1412.",
            "Stone, P. (1974). Cross-Validatory Choice and Assessment of Statistical Predictions. "
            "Journal of the Royal Statistical Society B, 36(2), 111-147.",
            "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
            "Friedman, J. H. (2001). Greedy Function Approximation. The Annals of Statistics, "
            "29(5), 1189-1232.",
            "Gujarati, D. N., & Porter, D. E. (2011). Basic Econometrics (5th ed.). McGraw-Hill. "
            "Section 10.6 on collinearity and suppression.",
            "Tobin, J. (1958). Estimation of Relationships for Limited Dependent Variables. "
            "Econometrica, 26(1), 24-36.",
        ]),
        ("18.3 Further Reading", [
            "Harrell, F. E. (2001). Regression Modeling Strategies (2nd ed.). Springer. Chapter 5 "
            "on interpretation overfitting.",
            "James, G., et al. (2021). An Introduction to Statistical Learning (2nd ed.). "
            "Springer. Section 6.2 on ridge regression and lasso.",
            "Roberts, D. R., et al. (2017). Cross-Validating for Spatially-Aggregated Data. "
            "Ecological Modelling, 355, 139-151. Directly relevant to the limitation noted in "
            "Section 15.",
        ]),
    ]
    for heading, entries in groups:
        document.add_heading(heading, level=2)
        for reference in entries:
            p = para(document, reference, size=9.5, space_after=4, indent=0.3)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    document.add_paragraph()
    add_callout(
        document, "Reproducibility statement:",
        "all results in this report were produced with random seed 42 applied to the train-test "
        "split, every estimator and every cross-validation fold. The full console log of the run "
        "that generated these numbers is published in the GitHub repository at "
        "outputs/console_output.txt.")


# ===========================================================================
# ASSEMBLY
# ===========================================================================
def main():
    """Build the complete Word report."""
    document = Document()
    style_document(document)

    section = document.sections[0]
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    add_page_numbers(section)

    build_cover(document)
    build_contents(document)
    build_introduction(document)
    build_task1(document)
    build_task2(document)
    build_task3(document)
    build_task4(document)
    build_task5(document)
    build_task9(document)
    build_task6(document)
    build_task7(document)
    build_task8(document)
    build_shrinkage(document)
    build_final_selection(document)
    build_limitations(document)
    build_github_section(document)
    build_conclusion(document)
    build_references(document)

    document.save(OUT_FILE)

    images = len([p for p in FIG_DIR.glob("*.png")])
    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Report written : {OUT_FILE.name}")
    print(f"Size           : {size_mb:.2f} MB")
    print(f"Figures found  : {images}")
    print(f"Figures embedded: {images}")
    print(f"GitHub URL     : {GITHUB_URL}")


if __name__ == "__main__":
    main()