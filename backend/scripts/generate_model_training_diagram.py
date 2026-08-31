"""
generate_model_training_diagram.py — Generates a clean, retro-style model training flow diagram
illustrating the AXE Genesis 2-Tier Training & Execution Concept from kaggle_axe_meta_learner_training.ipynb.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = "/home/daniel-joseph/.gemini/antigravity/brain/d08ddca1-cfd9-4d86-846c-eb6a59907032"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "retro_model_training_diagram.png")

def generate_training_diagram():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Color Scheme (Retro / Clean High Contrast)
    BORDER_COLOR = "#1E293B"
    TEXT_COLOR = "#0F172A"
    SUBTEXT_COLOR = "#475569"

    # Row 1: Data Preprocessing & Features
    box_data = {"x": 0.5, "y": 6.2, "w": 3.2, "h": 1.4, "title": "1. Multi-Timeframe Feature Matrix", "sub": "40,000 Bars (5m LTF + DXY Context)\n1000-Bar Lookback x N Features", "bg": "#EFF6FF", "border": "#2563EB"}
    
    # Row 1: Tier 1 Meta-Learner
    box_tier1 = {"x": 4.4, "y": 6.2, "w": 3.4, "h": 1.4, "title": "2. Tier 1: Multi-Task Meta-Learner", "sub": "Conv1D/Dense + 6 Auxiliary Heads\nOutputs: Conviction, Expiry (5m-1h), MFE/MAE", "bg": "#F0FDF4", "border": "#16A34A"}

    # Row 1: Zero-Lookahead SNR Zone Engine
    box_snr = {"x": 8.4, "y": 6.2, "w": 3.1, "h": 1.4, "title": "3. 0% Lookahead SNR Engine", "sub": "Vectorized Sequential K-Means\nCausal Window (df.iloc[:idx+1])", "bg": "#FEF3C7", "border": "#D97706"}

    # Row 2: Sequential Traversal & Hard Action Mask
    box_mask = {"x": 0.5, "y": 3.6, "w": 3.2, "h": 1.4, "title": "4. Zone HardActionMask", "sub": "ATR Proximity & Volume Delta\nStrict No-Chase Entry Filtering", "bg": "#FEE2E2", "border": "#DC2626"}

    # Row 2: Tier 2 Q-Executor Traversal Training
    box_tier2 = {"x": 4.4, "y": 3.6, "w": 3.4, "h": 1.4, "title": "5. Tier 2: Q-Executor Traversal", "sub": "Sequential Path Walk (Account State)\n28-Dim State + DeepScalper Rewards", "bg": "#F3E8FF", "border": "#9333EA"}

    # Row 2: Out-of-Sample Holdout Evaluation
    box_eval = {"x": 8.4, "y": 3.6, "w": 3.1, "h": 1.4, "title": "6. Out-of-Sample Walk-Forward", "sub": "8,000 Holdout Test Bars (20% Split)\nExpiry Alignment & Sharpe Metric", "bg": "#ECFEFF", "border": "#0891B2"}

    # Row 3: Final Autonomous Production Agent
    box_agent = {"x": 2.5, "y": 0.8, "w": 7.0, "h": 1.5, "title": "7. Production Checkpoint Bundle & Alpaca Live Trading Agent", "sub": "Hydrates meta_learner_best.pt & q_executor_best.pt → Alpaca REST API & CLI Wrapper\nAutonomous Cycle: Real-time DXY/MTF scoring → Zone-anchored Options order execution", "bg": "#F1F5F9", "border": "#0F172A"}

    all_boxes = [box_data, box_tier1, box_snr, box_mask, box_tier2, box_eval, box_agent]

    for b in all_boxes:
        rect = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"],
                                      boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor=b["bg"], edgecolor=b["border"], linewidth=1.8)
        ax.add_patch(rect)
        
        # Adjust text sizing for large agent box vs standard boxes
        title_size = 10 if b != box_agent else 11
        sub_size = 7.5 if b != box_agent else 8.5
        
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.70, b["title"],
                color=TEXT_COLOR, fontsize=title_size, fontweight="bold", ha="center")
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.32, b["sub"],
                color=SUBTEXT_COLOR, fontsize=sub_size, ha="center", multialignment="center")

    # Connectors (Clean retro arrow lines)
    arrow_style = dict(arrowstyle="-|>", color="#334155", lw=1.8, mutation_scale=12)

    # Row 1 Horizontal Connectors: Data -> Tier 1 -> SNR
    ax.annotate("", xy=(4.4, 6.9), xytext=(3.7, 6.9), arrowprops=arrow_style)
    ax.annotate("", xy=(8.4, 6.9), xytext=(7.8, 6.9), arrowprops=arrow_style)

    # Downward Connectors to Row 2
    ax.annotate("", xy=(2.1, 5.0), xytext=(2.1, 6.2), arrowprops=arrow_style) # Data to Mask
    ax.annotate("", xy=(6.1, 5.0), xytext=(6.1, 6.2), arrowprops=arrow_style) # Tier 1 to Tier 2
    ax.annotate("", xy=(9.95, 5.0), xytext=(9.95, 6.2), arrowprops=arrow_style) # SNR to Eval

    # Row 2 Horizontal Connectors: Mask -> Tier 2 -> Eval
    ax.annotate("", xy=(4.4, 4.3), xytext=(3.7, 4.3), arrowprops=arrow_style)
    ax.annotate("", xy=(8.4, 4.3), xytext=(7.8, 4.3), arrowprops=arrow_style)

    # Downward Connectors from Row 2 to Agent Box
    ax.annotate("", xy=(6.0, 2.3), xytext=(6.0, 3.6), arrowprops=arrow_style)

    ax.set_title("AXE Genesis Model Training Concept & Autonomous Execution Pipeline",
                 color="#0F172A", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated retro model training diagram at: {OUTPUT_PATH}")
    return OUTPUT_PATH

if __name__ == "__main__":
    generate_training_diagram()
