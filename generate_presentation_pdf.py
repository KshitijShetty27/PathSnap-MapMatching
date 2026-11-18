import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)
PDF_PATH = os.path.join(REPORT_DIR, "PathSnap_Review1_Presentation_Guide.pdf")


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "PathSnap: Review 1 Presentation & Evaluation Guide | PES University")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "UE24CS352A — Machine Learning | CSE Dept, PES University")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=15
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=3
    )
    code_box = ParagraphStyle(
        "CodeBox",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1A202C"),
        backColor=colors.HexColor("#F7FAFC"),
        borderColor=colors.HexColor("#E2E8F0"),
        borderWidth=1,
        borderPadding=6,
        spaceAfter=6
    )

    story = []

    # Title block
    story.append(Paragraph("PathSnap: Review 1 Presentation Guide", title_style))
    story.append(Paragraph("<b>UE24CS352A — Machine Learning</b> | Department of CSE, PES University<br/><b>Team:</b> Kshitij Satish Shetty (PES1UG24AM143) | Krrish Singla (PES1UG24AM141) | Kota Videesha (PES1UG24AM140) | Kshama (PES1UG24AM142)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A365D"), spaceAfter=12))

    # Section 1: Live Demo & Screenshot Breakdown
    story.append(Paragraph("1. Detailed Breakdown of Your Live Demo Screenshots", h1_style))
    story.append(Paragraph(
        "Your live setup successfully connected Laptop 1 (Backend) and Laptop 2 (Client). Here is the exact technical explanation of every metric and log line visible in your screenshots:",
        body_style
    ))

    # Screenshot 1 Breakdown
    story.append(Paragraph("A. Laptop 1 Backend Console Output (FastAPI Terminal)", h2_style))
    console_data = [
        [Paragraph("<b>Log Line from Console</b>", body_style), Paragraph("<b>Engineering Meaning to Tell the Evaluator</b>", body_style)],
        [Paragraph("<code>IPv4 Address: 10.8.120.105:8000</code>", code_box), Paragraph("The local private Wi-Fi socket on Laptop 1 hosting the model backend.", body_style)],
        [Paragraph("<code>Loaded 6,854 GPS observations, 1,571 nodes, 3,225 arcs</code>", code_box), Paragraph("Raw track data for trip 00000015 has 6,854 individual GPS points spanning a road subgraph of 1,571 intersections and 3,225 directed road arcs.", body_style)],
        [Paragraph("<code>Spatial candidate indexing completed in 77.5ms</code>", code_box), Paragraph("Discretized 3-point arcs queried via <code>cKDTree</code> in under 78 milliseconds, avoiding brute-force O(N*M) distance computation.", body_style)],
        [Paragraph("<code>Extracted 9 features across 68,540 candidate segments</code>", code_box), Paragraph("For each of the 6,854 GPS points, 10 road candidate segments are scored (6,854 * 10 = 68,540 feature rows) with 9 geometric/motion features.", body_style)],
        [Paragraph("<code>ML Model inference completed in 1852.7ms</code>", code_box), Paragraph("Tuned HistGradientBoosting scores all 68,540 candidates in 1.85 seconds (0.027 ms per observation), outputting calibrated emission probabilities.", body_style)],
        [Paragraph("<code>HMM / Viterbi decoding completed in 369.8ms</code>", code_box), Paragraph("Dynamic programming Viterbi algorithm reconstructs the globally optimal sequence across all 6,854 time steps in 370 ms.", body_style)],
        [Paragraph("<code>Top-1 Accuracy: 59.53% | Continuity: 100.00%</code>", code_box), Paragraph("Even in dense forest terrain with severe GPS multipath noise, Viterbi forces 100% legal road adjacency, eliminating disconnected teleportation.", body_style)]
    ]
    t_console = Table(console_data, colWidths=[200, 304])
    t_console.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_console)
    story.append(Spacer(1, 8))

    # Screenshot 2 Breakdown
    story.append(Paragraph("B. Laptop 2 Frontend Dashboard (Streamlit UI)", h2_style))
    ui_data = [
        [Paragraph("<b>UI Metric Card</b>", body_style), Paragraph("<b>Value in Screenshot</b>", body_style), Paragraph("<b>How to Explain This Metric to Professors</b>", body_style)],
        [Paragraph("<b>Route Continuity</b>", body_style), Paragraph("<b>100.0%</b> (+8.6% vs Clf)", body_style), Paragraph("Every single consecutive arc in the reconstructed path shares a physical junction in the road graph. Zero teleportations.", body_style)],
        [Paragraph("<b>Top-1 Accuracy (HMM)</b>", body_style), Paragraph("<b>59.53%</b> (+4.4% vs Clf)", body_style), Paragraph("Observation-level match against ground truth. Notice that HMM decoding increased raw classifier accuracy from 55.11% to 59.53%.", body_style)],
        [Paragraph("<b>Classifier-Only Acc</b>", body_style), Paragraph("<b>55.11%</b>", body_style), Paragraph("Accuracy if the ML model acts independently without topological road graph constraints.", body_style)],
        [Paragraph("<b>Sequence Recovery</b>", body_style), Paragraph("<b>41.72%</b>", body_style), Paragraph("Jaccard sequence similarity between unique road arcs traversed and reference route.", body_style)],
        [Paragraph("<b>Total Latency</b>", body_style), Paragraph("<b>3,319.6 ms</b>", body_style), Paragraph("Complete end-to-end pipeline time for nearly 7,000 observations (~0.48 ms per GPS point).", body_style)]
    ]
    t_ui = Table(ui_data, colWidths=[120, 110, 274])
    t_ui.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_ui)
    story.append(Spacer(1, 8))

    # Trip 15 vs Trip 87/49 Explanation
    story.append(Paragraph("<b>Why did Trip 00000015 score 59.5% accuracy while Trip 00000087 scored 97.4%?</b>", h2_style))
    story.append(Paragraph(
        "Trip 00000015 is a long circular loop through the Meerdaalwoud forest park. Tree canopy foliage and winding trails create high lateral GPS drift where parallel tracks run meters apart. Despite the noisy telemetry, our HMM Viterbi decoder achieved <b>100.0% Route Continuity</b>. On urban and arterial trips like <b>Trip 00000087</b> and <b>Trip 00000049</b>, our pipeline achieves <b>97.36% and 95.99% accuracy</b> with <b>99.4% continuity</b>. Demonstrating both shows that the system generalizes under diverse real-world conditions.",
        body_style
    ))

    story.append(PageBreak())

    # Section 2: Complete Review 1 Agenda & Presentation Structure
    story.append(Paragraph("2. What and All to Know to Present (Core Agenda)", h1_style))
    story.append(Paragraph(
        "Walk the evaluators through the 4 mandatory phases in logical order:",
        body_style
    ))

    agenda_points = [
        "<b>Phase 1: Exploratory Data Analysis (EDA):</b> 80 train trips, 10 student-test trips, 10 reserved evaluator trips. Nominal 1.0 Hz sampling (96.94% exactly 1s apart). Converted raw WGS84 angular degrees into local metric Cartesian meters using Equirectangular projection centered at trip mean latitude.",
        "<b>Phase 2: Spatial Indexing & Candidate Generation:</b> Road network has 165,790 directed arcs. Each arc is discretized at 20%, 50%, and 80% fractions and indexed via <code>cKDTree</code> (42 ms query time, 98.25% candidate recall at k=5). Note the 9.24% positive class imbalance.",
        "<b>Phase 3: Feature Engineering & Selection (9 Features):</b> Extracted perpendicular distance (r = -0.428), heading error (r = -0.312), normalized distance, speed difference, graph shortest path distance, arc length, network connectivity, trajectory curvature, and projection ratio t.",
        "<b>Phase 4: 5 Machine Learning Algorithm Families:</b> Evaluated Logistic Regression (71.42%), KNN (69.00%), Decision Tree (70.62%), Random Forest (73.03%), and HistGradientBoosting (73.28%). HistGradientBoosting won due to highest Top-1 accuracy, lowest log loss (0.1243), and 11.6x faster inference than Random Forest.",
        "<b>Phase 5: Bayesian Hyperparameter Optimization:</b> Optuna TPE over 20 trials optimizing 5-fold GroupKFold accuracy. Found optimal max_iter=140, learning_rate=0.0456, max_depth=7, min_samples_leaf=18, l2_reg=4.134. Validation accuracy reached 93.97%, test accuracy 73.41%.",
        "<b>Phase 6: Sequential Route Decoding (HMM Viterbi):</b> Point-level classifiers produce disjointed paths (88% continuity). Feeding calibrated emission probabilities into an HMM with graph transition penalties boosts continuity to <b>98.86% across all 10 evaluator trips</b>."
    ]
    for pt in agenda_points:
        story.append(Paragraph(f"• {pt}", bullet_style))

    story.append(Spacer(1, 10))

    # Section 3: Evaluator Benchmark Summary Table
    story.append(Paragraph("3. Reserved Evaluator Test Set Benchmark (10 Trips)", h1_style))
    eval_table_data = [
        ["Trip ID", "Observations", "Classifier Acc", "HMM Top-1 Acc", "HMM Continuity", "Sequence Jaccard"],
        ["00000015", "6,854", "55.11%", "59.53%", "100.00%", "41.72%"],
        ["00000017", "1,346", "85.14%", "82.76%", "96.65%", "79.33%"],
        ["00000023", "458", "84.06%", "86.02%", "97.59%", "60.22%"],
        ["00000030", "654", "75.99%", "76.91%", "97.40%", "61.36%"],
        ["00000049", "2,642", "94.47%", "95.99%", "99.47%", "87.69%"],
        ["00000059", "1,031", "88.75%", "86.61%", "98.25%", "78.99%"],
        ["00000067", "916", "70.74%", "74.34%", "100.00%", "75.70%"],
        ["00000073", "1,144", "88.55%", "94.58%", "100.00%", "87.78%"],
        ["00000087", "1,399", "90.71%", "97.36%", "99.36%", "97.39%"],
        ["00000094", "3,230", "73.50%", "80.53%", "99.85%", "79.57%"],
        ["Average", "1,967", "80.70%", "83.46%", "98.86%", "74.97%"]
    ]
    t_eval = Table(eval_table_data, colWidths=[70, 75, 85, 90, 94, 90])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#EDF2F7")),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 10))

    # Section 4: Individual Team Member Ownership & Script
    story.append(Paragraph("4. Individual Team Member Defense Scripts", h1_style))
    team_data = [
        [Paragraph("<b>Member & SRN</b>", body_style), Paragraph("<b>Assigned Module</b>", body_style), Paragraph("<b>Opening Pitch to Recite</b>", body_style)],
        [
            Paragraph("<b>Kshitij Satish Shetty</b><br/>PES1UG24AM143", body_style),
            Paragraph("Pipeline, Boosting Model & Bayesian Tuning", body_style),
            Paragraph("<i>'I orchestrated the zero-leakage training pipeline, implemented HistGradientBoosting with binned histograms, and conducted Bayesian optimization with Optuna TPE over 20 trials, finding regularized tree parameters that reached 93.97% CV accuracy.'</i>", body_style)
        ],
        [
            Paragraph("<b>Krrish Singla</b><br/>PES1UG24AM141", body_style),
            Paragraph("EDA & Coordinate Projection", body_style),
            Paragraph("<i>'I owned the exploratory data analysis across 80 training trips, verified the 1.0 Hz nominal sampling rate, and formulated the equirectangular projection to convert WGS84 angular degrees into local metric meters.'</i>", body_style)
        ],
        [
            Paragraph("<b>Kota Videesha</b><br/>PES1UG24AM140", body_style),
            Paragraph("Spatial Index & Feature Selection", body_style),
            Paragraph("<i>'I developed the 3-point arc discretization and cKDTree index (42 ms query, 98.25% recall) and engineered our 9 features, analyzing Pearson correlation with target Y.'</i>", body_style)
        ],
        [
            Paragraph("<b>Kshama</b><br/>PES1UG24AM142", body_style),
            Paragraph("Model Comparison & Baselines", body_style),
            Paragraph("<i>'I benchmarked Logistic Regression, KNN, Decision Tree, and Random Forest against HistGradientBoosting across all student test trips on accuracy, continuity, F1, log loss, and latency.'</i>", body_style)
        ]
    ]
    t_team = Table(team_data, colWidths=[100, 110, 294])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_team)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Wrote presentation guide PDF to: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
