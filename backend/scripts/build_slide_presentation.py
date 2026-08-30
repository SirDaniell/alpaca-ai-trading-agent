"""
build_slide_presentation.py — Generates a 6-slide landscape 16:9 PDF presentation
for the lablab.ai Alpaca AI Hackathon submission.
"""

import sys
import os
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
from reportlab.pdfgen import canvas

PDF_PATH = "/media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/lablab.ai Alpaca AI/alpaca_ai_options_rl_presentation.pdf"
COVER_IMAGE_PATH = "/home/daniel-joseph/.gemini/antigravity/brain/d08ddca1-cfd9-4d86-846c-eb6a59907032/trader_cat_cover_image_1788093539555.png"

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
        # Top banner line
        self.setStrokeColor(colors.HexColor("#00F0FF"))
        self.setLineWidth(2)
        self.line(30, 580, 762, 580)

        # Footer banner
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#8A99AD"))
        self.drawString(30, 20, "Alpaca AI Options RL Execution Pipeline — AXE Genesis Architecture")
        
        page_text = f"Slide {self._pageNumber} of {page_count}"
        self.drawRightString(762, 20, page_text)
        self.restoreState()


DIAGRAM_ARCH_PATH = "/home/daniel-joseph/.gemini/antigravity/brain/d08ddca1-cfd9-4d86-846c-eb6a59907032/architecture_flowchart_diagram_1788093907084.png"
DIAGRAM_REWARD_PATH = "/home/daniel-joseph/.gemini/antigravity/brain/d08ddca1-cfd9-4d86-846c-eb6a59907032/reward_shaping_diagram_1788093933518.png"

def create_presentation():
    # 16:9 / Landscape Letter Dimensions (792 x 612 pt)
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    BG_DARK = colors.HexColor("#0B0E14")
    PRIMARY_CYAN = colors.HexColor("#00F0FF")
    ACCENT_GREEN = colors.HexColor("#00E676")
    TEXT_LIGHT = colors.HexColor("#F0F4F8")
    SUBTEXT = colors.HexColor("#94A3B8")

    title_style = ParagraphStyle(
        "SlideTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=PRIMARY_CYAN,
        alignment=0,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "SlideSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=15,
        textColor=SUBTEXT,
        spaceAfter=12,
    )

    body_style = ParagraphStyle(
        "SlideBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=TEXT_LIGHT,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "SlideBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=TEXT_LIGHT,
        leftIndent=12,
        spaceAfter=5,
    )

    story = []

    # ── SLIDE 1: Title & Cover Slide ──────────────────────────────────────────
    story.append(Paragraph("Alpaca AI Options RL Execution Pipeline", title_style))
    story.append(Paragraph("Instrument-Agnostic Contextual Q-Learning & Multi-Horizon Meta-Learner (AXE Genesis Architecture)", subtitle_style))

    if os.path.exists(COVER_IMAGE_PATH):
        img_cover = Image(COVER_IMAGE_PATH, width=350, height=197)
        
        bullets = [
            Paragraph("<b>Core Goal:</b> Discipline-driven options execution with zero-lookahead SNR zones.", bullet_style),
            Paragraph("<b>Two-Tier Framework:</b> HTF Meta-Learner (Tier 1) + Dual-Branch Q-Executor (Tier 2).", bullet_style),
            Paragraph("<b>Parity Engine:</b> 1:1 functional PyTorch & Keras reinforcement learning parity.", bullet_style),
            Paragraph("<b>Reward Shaping:</b> DeepScalper hindsight bonus & 'Wise Patience' rewards.", bullet_style),
            Paragraph("<b>Cross-Symbol Universe:</b> Standardized USD ETF/Equity testing (GLD, SPY, QQQ, etc.).", bullet_style),
        ]
        
        t1 = Table([[img_cover, bullets]], colWidths=[360, 360])
        t1.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (1,0), (1,0), 10),
        ]))
        story.append(t1)
    
    story.append(PageBreak())

    # ── SLIDE 2: Two-Tier Architecture & Flowchart Diagram ─────────────────────
    story.append(Paragraph("1. Two-Tier Architecture & Workflow Pipeline", title_style))
    story.append(Paragraph("Decoupled multi-horizon context scoring and zero-interference execution.", subtitle_style))

    arch_bullets = [
        Paragraph("<b>Tier 1 — Signal Meta-Learner:</b>", body_style),
        Paragraph("• Evaluates forward 24-bar price statistics (MFE, MAE, reversal timing) to output conviction score [0.0 - 1.0].", bullet_style),
        Paragraph("• Predicts optimal option expiry duration (5m, 15m, 30m, 1h) based on MTF RSI & DXY regime context.", bullet_style),
        Paragraph("• Features 6 decoupled auxiliary regression heads with private projections for zero gradient interference.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>Tier 2 — Dual-Branch Ensemble Options Q-Learner:</b>", body_style),
        Paragraph("• Microstructure + MTF Alignment gated fusion layer.", bullet_style),
        Paragraph("• 5-action space: <code>WAIT</code>, <code>CALL</code>, <code>PUT</code>, <code>TP_HALF</code>, <code>CLOSE</code>.", bullet_style),
        Paragraph("• Full 1:1 functional parity between PyTorch and Keras.", bullet_style),
    ]

    if os.path.exists(DIAGRAM_ARCH_PATH):
        img_arch = Image(DIAGRAM_ARCH_PATH, width=340, height=272)
        t_arch = Table([[arch_bullets, img_arch]], colWidths=[370, 350])
        t_arch.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('RIGHTPADDING', (0,0), (0,0), 10),
        ]))
        story.append(t_arch)
    else:
        story.extend(arch_bullets)

    story.append(PageBreak())

    # ── SLIDE 3: Reward Shaping & Infographic Diagram ─────────────────────────
    story.append(Paragraph("2. DeepScalper Hindsight Reward Shaping & Wise Patience", title_style))
    story.append(Paragraph("Aligning execution discipline against opportunity cost without distortion.", subtitle_style))

    reward_text = [
        Paragraph("<b>Key Reward Components:</b>", body_style),
        Paragraph("• <b>Wise Patience Bonus (+0.15):</b> Rewards <code>WAIT</code> when standing down avoids adverse loss.", bullet_style),
        Paragraph("• <b>Missed Opportunity (-0.15):</b> Penalizes <code>WAIT</code> if an unmasked setup played out.", bullet_style),
        Paragraph("• <b>Best Price Entry (+0.15):</b> Rewards entries with minimal drawdown exposure (≤ 0.5%).", bullet_style),
        Paragraph("• <b>Discipline Bonus (+0.02):</b> Constant reward for <code>WAIT</code> during low-conviction regimes.", bullet_style),
        Paragraph("• <b>No-Chase Masking:</b> Enforces SNR proximity & volume delta before unmasking CALL/PUT.", bullet_style),
    ]

    if os.path.exists(DIAGRAM_REWARD_PATH):
        img_reward = Image(DIAGRAM_REWARD_PATH, width=340, height=272)
        t_rew_layout = Table([[reward_text, img_reward]], colWidths=[370, 350])
        t_rew_layout.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('RIGHTPADDING', (0,0), (0,0), 10),
        ]))
        story.append(t_rew_layout)
    else:
        story.extend(reward_text)

    story.append(PageBreak())

    # ── SLIDE 4: Data Integrity & Real Market Engine ──────────────────────────
    story.append(Paragraph("3. Data Integrity & Real Market Engine", title_style))
    story.append(Paragraph("Strict zero-lookahead S&R detection and dynamic instrument scale normalization.", subtitle_style))

    story.append(Paragraph("<b>Zero-Lookahead SNR Zone Detection:</b>", body_style))
    story.append(Paragraph("• Uses vectorized <code>support_resistance.py</code> module (<code>detect_snr_levels_sequential</code> + <code>create_clustered_zones_sequential</code>).", bullet_style))
    story.append(Paragraph("• Zone snapshots are generated dynamically per bar strictly using causal historical slices: <code>df.iloc[:idx+1]</code>.", bullet_style))
    story.append(Paragraph("• Completely eliminates lookahead bias and static quantile placeholders across all 4 evaluation phases.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Real Buyer/Seller Volume Profiles & No-Chase Masking:</b>", body_style))
    story.append(Paragraph("• <code>_make_exec_ctx</code> computes real candle volume profiles (buy volume vs sell volume) directly from market data.", bullet_style))
    story.append(Paragraph("• <code>HardActionMask</code> enforces zone proximity and volume delta confirmation before unmasking CALL/PUT entry actions.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Dynamic Instrument Metadata Normalization:</b>", body_style))
    story.append(Paragraph("• <code>get_instrument_metadata</code> automatically maps ETF/Equity underlyings (GLD, SPY, QQQ, TLT, SLV, AAPL, etc.) to <code>pip_size=0.01</code>.", bullet_style))
    story.append(Paragraph("• Eliminates 100x scale distortions between Equities ($0.01 = 1 cent = 1 pip) and FX/Crypto.", bullet_style))

    story.append(PageBreak())

    # ── SLIDE 5: Cross-Symbol Evaluation & Edge Validation ────────────────────
    story.append(Paragraph("4. Cross-Symbol Out-of-Sample Evaluation", title_style))
    story.append(Paragraph("Multi-symbol walk-forward backtesting across standardized option underlyings.", subtitle_style))

    story.append(Paragraph("<b>Cross-Symbol Benchmark Loop:</b>", body_style))
    story.append(Paragraph("• Evaluates 40,000 historical 5m bars (~2 years of market data) per symbol across ETF/Equity universe.", bullet_style))
    story.append(Paragraph("• Out-of-sample holdout test set (20% split, ~8,000 bars) walk-forward execution.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Scientific Notation Logging & Loss Convergence:</b>", body_style))
    story.append(Paragraph("• All training logs standardized to scientific notation (<code>:.4e</code>) to track micro-losses across auxiliary heads.", bullet_style))
    story.append(Paragraph("• Confirms loss convergence for Q-values, directional strength, pips proxy, and reversal risk heads.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Expiry Horizon Selection Summary:</b>", body_style))
    story.append(Paragraph("• Classifies optimal duration across 5m (1 bar), 15m (3 bars), 30m (6 bars), and 1h (12 bars) option expiries.", bullet_style))
    story.append(Paragraph("• Reports directional win rates, maximum win/loss streak bounds, and meta-learner expiry alignment rates.", bullet_style))

    story.append(PageBreak())

    # ── SLIDE 6: Summary & Hackathon Highlights ───────────────────────────────
    story.append(Paragraph("5. Summary & Key System Highlights", title_style))
    story.append(Paragraph("Robust, production-grade RL execution architecture built for Alpaca AI.", subtitle_style))

    summary_box_data = [
        [Paragraph("<b>Key Achievement</b>", body_style), Paragraph("<b>Implementation Details</b>", body_style)],
        [Paragraph("<b>Two-Tier RL Architecture</b>", body_style), Paragraph("Tier 1 Meta-Learner (Context/Expiry) + Tier 2 Dual-Branch Q-Executor (Entry/Exit).", body_style)],
        [Paragraph("<b>PyTorch / Keras Parity</b>", body_style), Paragraph("100% architectural & reward parity across both major deep learning frameworks.", body_style)],
        [Paragraph("<b>DeepScalper Reward Shaping</b>", body_style), Paragraph("Wise patience bonuses and hindsight opportunity penalties to train disciplined policy.", body_style)],
        [Paragraph("<b>0% Lookahead SNR Engine</b>", body_style), Paragraph("Sequential K-Means volume profile zone snapshots operating on causal price slices.", body_style)],
        [Paragraph("<b>Alpaca Market Pipeline</b>", body_style), Paragraph("Real-time MTF bar fetching with hard timeout guards and dynamic DXY inversion features.", body_style)],
    ]

    t_sum = Table(summary_box_data, colWidths=[200, 520])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('TEXTCOLOR', (0,0), (-1,0), PRIMARY_CYAN),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#334155")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(t_sum)

    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Submitted for lablab.ai Alpaca AI Hackathon — Built with PyTorch, Keras, Alpaca API & Vanilla CSS.</b>", ParagraphStyle("SubTextCenter", parent=subtitle_style, alignment=1, textColor=ACCENT_GREEN)))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF slide presentation at: {PDF_PATH}")

if __name__ == "__main__":
    create_presentation()
