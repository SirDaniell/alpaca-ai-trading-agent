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
COVER_IMAGE_PATH = "/home/daniel-joseph/.gemini/antigravity/brain/d08ddca1-cfd9-4d86-846c-eb6a59907032/retro_candlestick_chart.png"
DIAGRAM_ARCH_PATH = "/home/daniel-joseph/.gemini/antigravity/brain/d08ddca1-cfd9-4d86-846c-eb6a59907032/retro_architecture_flowchart.png"
DIAGRAM_REWARD_PATH = "/home/daniel-joseph/.gemini/antigravity/brain/d08ddca1-cfd9-4d86-846c-eb6a59907032/retro_reward_shaping_chart.png"
DIAGRAM_TRAINING_PATH = "/home/daniel-joseph/.gemini/antigravity/brain/d08ddca1-cfd9-4d86-846c-eb6a59907032/retro_model_training_diagram.png"

def draw_page_background(canvas_obj, doc_obj):
    canvas_obj.saveState()
    canvas_obj.setFillColor(colors.HexColor("#0B0E14"))
    canvas_obj.rect(0, 0, 792, 612, fill=True, stroke=False)
    canvas_obj.restoreState()

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
        # Top cyan accent line
        self.setStrokeColor(colors.HexColor("#00F0FF"))
        self.setLineWidth(2)
        self.line(30, 580, 762, 580)

        # Footer text
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#8A99AD"))
        self.drawString(30, 20, "Alpaca AI Options RL Execution Pipeline — AXE Genesis Architecture")
        
        page_text = f"Slide {self._pageNumber} of {page_count}"
        self.drawRightString(762, 20, page_text)
        self.restoreState()

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
    story.append(Paragraph("AXE Genesis — Alpaca AI Options RL Pipeline", title_style))
    story.append(Paragraph("DXY Divergence Alpha, Multi-Horizon Meta-Learner & Contextual Q-Learning Execution", subtitle_style))

    if os.path.exists(COVER_IMAGE_PATH):
        img_cover = Image(COVER_IMAGE_PATH, width=350, height=197)
        
        bullets = [
            Paragraph("<b>Core Strategy:</b> Exploiting Dollar Index (DXY) vs Asset divergence at key SNR zones.", bullet_style),
            Paragraph("<b>Two-Tier Framework:</b> HTF Meta-Learner (Tier 1) + Dual-Branch Q-Executor (Tier 2).", bullet_style),
            Paragraph("<b>Public Repository:</b> <code>https://github.com/SirDaniell/alpaca-ai-trading-agent</code>", bullet_style),
            Paragraph("<b>Framework Parity:</b> 1:1 functional PyTorch & Keras reinforcement learning models.", bullet_style),
            Paragraph("<b>Reward Shaping:</b> DeepScalper hindsight bonus, 'Wise Patience' & zero-chase action masking.", bullet_style),
            Paragraph("<b>Autonomous API/CLI:</b> Direct execution via Alpaca Trading API & Alpaca CLI wrapper.", bullet_style),
        ]
        
        t1 = Table([[img_cover, bullets]], colWidths=[360, 360])
        t1.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (1,0), (1,0), 10),
        ]))
        story.append(t1)
    
    story.append(PageBreak())

    # ── SLIDE 2: Core Edge — DXY Divergence & SNR Zones ─────────────────────
    story.append(Paragraph("1. Primary Trading Edge: DXY Divergence & Zone Anchoring", title_style))
    story.append(Paragraph("Capturing structural momentum shifts when underlying asset price diverges from the US Dollar Index.", subtitle_style))

    edge_bullets = [
        Paragraph("<b>Dollar Index (DXY) Macro Divergence:</b>", body_style),
        Paragraph("• Tracks real-time inverse correlation between DXY basket (EURUSD, USDJPY, GBPUSD, USDCAD) and traded underlying.", bullet_style),
        Paragraph("• Identifies bullish divergence (Asset making higher low while DXY fails to break higher) and bearish divergence.", bullet_style),
        Paragraph("• Divergence signals act as high-probability macro catalysts to filter out false breakouts.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>Zone-Anchored No-Chase Execution:</b>", body_style),
        Paragraph("• Causal 0%-lookahead Support & Resistance detection calculates real-time zone boundaries.", bullet_style),
        Paragraph("• <code>HardActionMask</code> strictly prohibits BUY_CALL / BUY_PUT unless price is within ATR proximity of a confirmed zone.", bullet_style),
        Paragraph("• Volume profile delta (buyers vs sellers) validates breakout/reversal before unmasking entry actions.", bullet_style),
    ]

    if os.path.exists(DIAGRAM_ARCH_PATH):
        img_arch = Image(DIAGRAM_ARCH_PATH, width=340, height=272)
        t_arch = Table([[edge_bullets, img_arch]], colWidths=[370, 350])
        t_arch.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('RIGHTPADDING', (0,0), (0,0), 10),
        ]))
        story.append(t_arch)
    else:
        story.extend(edge_bullets)

    story.append(PageBreak())

    # ── SLIDE 3: Two-Tier Architecture & Flowchart Diagram ─────────────────────
    story.append(Paragraph("2. Two-Tier Reinforcement Learning Architecture", title_style))
    story.append(Paragraph("Decoupled multi-horizon context scoring and zero-interference execution.", subtitle_style))

    arch_bullets = [
        Paragraph("<b>Tier 1 — Signal Meta-Learner:</b>", body_style),
        Paragraph("• Evaluates forward price statistics (MFE, MAE, reversal timing) to output conviction score [0.0 - 1.0].", bullet_style),
        Paragraph("• Predicts optimal option expiry duration (5m, 15m, 30m, 1h) based on MTF RSI & DXY regime context.", bullet_style),
        Paragraph("• Features 6 decoupled auxiliary regression heads with private projections for zero gradient interference.", bullet_style),
        Spacer(1, 6),
        Paragraph("<b>Tier 2 — Dual-Branch Ensemble Options Q-Learner:</b>", body_style),
        Paragraph("• Microstructure + MTF Alignment gated fusion layer.", bullet_style),
        Paragraph("• 5-action space: <code>WAIT</code>, <code>CALL</code>, <code>PUT</code>, <code>TP_HALF</code>, <code>CLOSE</code>.", bullet_style),
        Paragraph("• Full 1:1 functional parity between PyTorch and Keras.", bullet_style),
    ]

    if os.path.exists(DIAGRAM_REWARD_PATH):
        img_reward = Image(DIAGRAM_REWARD_PATH, width=340, height=272)
        t_rew_layout = Table([[arch_bullets, img_reward]], colWidths=[370, 350])
        t_rew_layout.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('RIGHTPADDING', (0,0), (0,0), 10),
        ]))
        story.append(t_rew_layout)
    else:
        story.extend(arch_bullets)

    story.append(PageBreak())

    # ── SLIDE 4: Reward Shaping & Data Integrity ──────────────────────────
    story.append(Paragraph("3. DeepScalper Reward Shaping & Data Integrity", title_style))
    story.append(Paragraph("Aligning execution discipline against opportunity cost with zero-lookahead guarantees.", subtitle_style))

    story.append(Paragraph("<b>DeepScalper Hindsight Reward Shaping:</b>", body_style))
    story.append(Paragraph("• <b>Wise Patience Bonus (+0.15):</b> Rewards <code>WAIT</code> when standing down avoids adverse drawdown.", bullet_style))
    story.append(Paragraph("• <b>Missed Opportunity (-0.15):</b> Penalizes <code>WAIT</code> if an unmasked setup played out successfully.", bullet_style))
    story.append(Paragraph("• <b>Best Price Entry (+0.15):</b> Rewards entries with minimal drawdown exposure (≤ 0.5%).", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Zero-Lookahead Causal SNR Zone Detection:</b>", body_style))
    story.append(Paragraph("• Vectorized <code>detect_snr_levels_sequential</code> ensures zone snapshots are generated strictly on <code>df.iloc[:idx+1]</code>.", bullet_style))
    story.append(Paragraph("• Completely eliminates lookahead leakage across historical training, Kaggle GPU pipelines, and live loops.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Dynamic Instrument Scaling:</b>", body_style))
    story.append(Paragraph("• <code>get_instrument_metadata</code> automatically maps Equities/ETFs (SPY, QQQ, GLD, AAPL) to $0.01 = 1 cent = 1 pip.", bullet_style))
    story.append(Paragraph("• Eliminates 100x scale distortions between Equities and FX/Crypto.", bullet_style))

    story.append(PageBreak())

    # ── SLIDE 5: Cross-Symbol Evaluation & Edge Validation ────────────────────
    story.append(Paragraph("4. Cross-Symbol Out-of-Sample Benchmark", title_style))
    story.append(Paragraph("Multi-symbol walk-forward backtesting across standardized option underlyings.", subtitle_style))

    story.append(Paragraph("<b>Cross-Symbol Benchmark Loop:</b>", body_style))
    story.append(Paragraph("• Evaluates 40,000 historical 5m bars (~2 years of market data) per symbol across ETF/Equity universe.", bullet_style))
    story.append(Paragraph("• Out-of-sample holdout test set (20% split, ~8,000 bars) walk-forward execution.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Multi-Head Loss Convergence:</b>", body_style))
    story.append(Paragraph("• Multi-task auxiliary heads (pips, risk, liquidity, reversal) use isolated gradient paths (<code>feat.detach()</code>).", bullet_style))
    story.append(Paragraph("• Prevents auxiliary loss spikes from corrupting primary directional Q-learning policies.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Expiry Duration Optimization:</b>", body_style))
    story.append(Paragraph("• Dynamically selects contract expiries across 5m (1 bar), 15m (3 bars), 30m (6 bars), and 1h (12 bars).", bullet_style))
    story.append(Paragraph("• Maximizes Sharpe ratio and win rate by matching expiry to predicted trend momentum.", bullet_style))

    story.append(PageBreak())

    # ── SLIDE 6: Training Concept & Pipeline Workflow ─────────────────────────
    story.append(Paragraph("5. Model Training Concept & Pipeline Workflow", title_style))
    story.append(Paragraph("4-Phase Kaggle GPU Pipeline: Multi-Task Meta-Learning + Sequential Q-Traversal.", subtitle_style))

    training_text = [
        Paragraph("<b>Training Concept Highlights:</b>", body_style),
        Paragraph("• <b>Phase 1 Multi-Task Meta-Learner:</b> 50 epochs over 40,000 bars using Cosine Annealing (1e-3 → 1e-5). Decoupled auxiliary heads eliminate gradient interference.", bullet_style),
        Paragraph("• <b>Phase 2 Sequential Traversal:</b> Sequential walk preserves path-dependent account state (drawdown %, streaks, open positions).", bullet_style),
        Paragraph("• <b>Phase 3 & 4 Out-of-Sample Eval:</b> Walk-forward backtest on 20% holdout bars (~8,000 bars) across 5m, 15m, 30m, 1h options.", bullet_style),
        Paragraph("• <b>Phase 5 Hydration Checkpoints:</b> Exports <code>meta_learner_best.pt</code> and <code>q_executor_best.pt</code> directly into backend engine.", bullet_style),
    ]

    if os.path.exists(DIAGRAM_TRAINING_PATH):
        img_train = Image(DIAGRAM_TRAINING_PATH, width=350, height=210)
        t_tr_layout = Table([[training_text, img_train]], colWidths=[350, 370])
        t_tr_layout.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('RIGHTPADDING', (0,0), (0,0), 10),
        ]))
        story.append(t_tr_layout)
    else:
        story.extend(training_text)

    story.append(PageBreak())

    # ── SLIDE 7: Summary & Submission Highlights ───────────────────────────────
    story.append(Paragraph("6. Summary & Hackathon Highlights", title_style))
    story.append(Paragraph("Production-grade autonomous options trading agent built on Alpaca infrastructure.", subtitle_style))

    table_header_style = ParagraphStyle("TableHeader", parent=body_style, fontName="Helvetica-Bold", textColor=PRIMARY_CYAN)
    table_cell_style = ParagraphStyle("TableCell", parent=body_style, textColor=colors.HexColor("#F8FAFC"))

    summary_box_data = [
        [Paragraph("<b>Key Feature</b>", table_header_style), Paragraph("<b>Implementation Details</b>", table_header_style)],
        [Paragraph("<b>Primary Alpha Edge</b>", table_cell_style), Paragraph("DXY vs Underlying divergence combined with zero-lookahead SNR zone anchoring.", table_cell_style)],
        [Paragraph("<b>Two-Tier RL Architecture</b>", table_cell_style), Paragraph("Tier 1 Meta-Learner (Expiry & Conviction) + Tier 2 Dual-Branch Q-Executor (Entry & Risk).", table_cell_style)],
        [Paragraph("<b>Alpaca Integration</b>", table_cell_style), Paragraph("Full Alpaca Trading API REST client + Alpaca CLI wrapper for order & position tracking.", table_cell_style)],
        [Paragraph("<b>Public Repository</b>", table_cell_style), Paragraph("<code>https://github.com/SirDaniell/alpaca-ai-trading-agent</code>", table_cell_style)],
        [Paragraph("<b>PyTorch / Keras Parity</b>", table_cell_style), Paragraph("Complete dual-framework implementation for maximum research flexibility.", table_cell_style)],
    ]

    t_sum = Table(summary_box_data, colWidths=[200, 520])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#0F172A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#334155")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(t_sum)

    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Submitted for lablab.ai Alpaca AI Hackathon — Built with PyTorch, Keras, Alpaca API & CLI.</b>", ParagraphStyle("SubTextCenter", parent=subtitle_style, alignment=1, textColor=ACCENT_GREEN)))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas, onFirstPage=draw_page_background, onLaterPages=draw_page_background)
    print(f"Successfully generated PDF slide presentation at: {PDF_PATH}")

if __name__ == "__main__":
    create_presentation()
