"""
generate_retro_charts.py — Generates clean, retro/classic 2D technical charts and flowcharts
using Matplotlib for the PDF slide presentation. No AI gradients or glow.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

CHARTS_DIR = "/home/daniel-joseph/.gemini/antigravity/brain/d08ddca1-cfd9-4d86-846c-eb6a59907032"

# ── 1. Retro Candlestick Chart with Buy/Sell Signals ──────────────────────────
def generate_candlestick_chart():
    np.random.seed(42)
    n_bars = 30
    
    # Generate synthetic price data
    price = 100.0
    opens, highs, lows, closes = [], [], [], []
    for _ in range(n_bars):
        change = np.random.normal(0.2, 1.2)
        o = price
        c = o + change
        h = max(o, c) + abs(np.random.normal(0.4, 0.3))
        l = min(o, c) - abs(np.random.normal(0.4, 0.3))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        price = c

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    fig.patch.set_facecolor('#1E293B')
    ax.set_facecolor('#0F172A')

    # Draw Candlesticks
    for i in range(n_bars):
        col = '#10B981' if closes[i] >= opens[i] else '#EF4444' # Green / Red
        # Wick
        ax.plot([i, i], [lows[i], highs[i]], color=col, linewidth=1.5)
        # Body
        body_bottom = min(opens[i], closes[i])
        body_height = max(abs(closes[i] - opens[i]), 0.05)
        rect = patches.Rectangle((i - 0.3, body_bottom), 0.6, body_height, facecolor=col, edgecolor=col)
        ax.add_patch(rect)

    # Add Moving Averages
    ma5 = [np.mean(closes[max(0, i-4):i+1]) for i in range(n_bars)]
    ma10 = [np.mean(closes[max(0, i-9):i+1]) for i in range(n_bars)]
    ax.plot(range(n_bars), ma5, color='#38BDF8', linewidth=1.5, label='MA 5')
    ax.plot(range(n_bars), ma10, color='#F59E0B', linewidth=1.5, label='MA 10')

    # Add Buy / Sell Signals
    buy_indices = [5, 18]
    sell_indices = [12, 25]

    for idx in buy_indices:
        ax.annotate('BUY', xy=(idx, lows[idx] - 0.8), xytext=(idx, lows[idx] - 2.2),
                    arrowprops=dict(facecolor='#10B981', edgecolor='#10B981', shrink=0.05, width=2, headwidth=7),
                    ha='center', va='top', fontsize=9, fontweight='bold', color='#10B981',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#064E3B', edgecolor='#10B981', alpha=0.9))

    for idx in sell_indices:
        ax.annotate('SELL', xy=(idx, highs[idx] + 0.8), xytext=(idx, highs[idx] + 2.2),
                    arrowprops=dict(facecolor='#EF4444', edgecolor='#EF4444', shrink=0.05, width=2, headwidth=7),
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#EF4444',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#7F1D1D', edgecolor='#EF4444', alpha=0.9))

    ax.set_title('Alpaca AI Options Execution — 5m LTF Signal Engine', color='#F8FAFC', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Historical 5m Bars', color='#94A3B8', fontsize=9)
    ax.set_ylabel('Underlying Price ($)', color='#94A3B8', fontsize=9)
    ax.tick_params(colors='#94A3B8', labelsize=8)
    ax.grid(True, color='#334155', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC', fontsize=8, loc='upper left')

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "retro_candlestick_chart.png")
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Generated retro_candlestick_chart.png")
    return path


# ── 2. Retro Architecture Flowchart ───────────────────────────────────────────
def generate_architecture_flowchart():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Draw Flowchart Boxes
    boxes = [
        {"x": 0.5, "y": 4.2, "w": 2.6, "h": 1.2, "title": "1. Alpaca Market Feed", "sub": "5m LTF Bars + DXY Context", "bg": "#EFF6FF", "border": "#2563EB"},
        {"x": 3.7, "y": 4.2, "w": 2.6, "h": 1.2, "title": "2. Tier 1 Meta-Learner", "sub": "Expiry Horizon Scoring (5m-1h)", "bg": "#F0FDF4", "border": "#16A34A"},
        {"x": 6.9, "y": 4.2, "w": 2.6, "h": 1.2, "title": "3. 0% Lookahead SNR", "sub": "Volume Profile & Zone Snapshots", "bg": "#FEF3C7", "border": "#D97706"},
        
        {"x": 2.1, "y": 1.2, "w": 2.6, "h": 1.2, "title": "4. Hard Action Mask", "sub": "No-Chase & SNR Confirmation", "bg": "#FEE2E2", "border": "#DC2626"},
        {"x": 5.3, "y": 1.2, "w": 2.6, "h": 1.2, "title": "5. Tier 2 Q-Executor", "sub": "Dual-Branch PyTorch/Keras", "bg": "#F3E8FF", "border": "#9333EA"},
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"], boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor=b["bg"], edgecolor=b["border"], linewidth=1.8)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.68, b["title"], color="#0F172A", fontsize=9.5, fontweight="bold", ha="center")
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.30, b["sub"], color="#475569", fontsize=7.5, ha="center")

    # Draw Connector Arrows
    arrow_style = dict(arrowstyle="-|>", color="#334155", lw=1.8, mutation_scale=12)

    # Top row connections
    ax.annotate("", xy=(3.7, 4.8), xytext=(3.1, 4.8), arrowprops=arrow_style)
    ax.annotate("", xy=(6.9, 4.8), xytext=(6.3, 4.8), arrowprops=arrow_style)

    # Down to Hard Action Mask
    ax.annotate("", xy=(3.4, 2.4), xytext=(3.4, 4.2), arrowprops=arrow_style)
    ax.annotate("", xy=(3.4, 2.4), xytext=(8.2, 4.2), arrowprops=arrow_style)

    # Hard Action Mask -> Tier 2 Q-Executor
    ax.annotate("", xy=(5.3, 1.8), xytext=(4.7, 1.8), arrowprops=arrow_style)

    ax.set_title("AXE Genesis Two-Tier Reinforcement Learning Pipeline", color="#0F172A", fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "retro_architecture_flowchart.png")
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Generated retro_architecture_flowchart.png")
    return path


# ── 3. Retro Reward Shaping Bar Chart ─────────────────────────────────────────
def generate_reward_shaping_chart():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    components = ['Wise Patience\n(+0.15)', 'Best Price Entry\n(+0.15)', 'Discipline Bonus\n(+0.02)', 'Churn Penalty\n(-0.05)', 'Missed Opportunity\n(-0.15)']
    values = [0.15, 0.15, 0.02, -0.05, -0.15]
    colors_list = ['#16A34A', '#2563EB', '#0D9488', '#E11D48', '#DC2626']

    bars = ax.bar(components, values, color=colors_list, width=0.55, edgecolor='#1E293B', linewidth=1)

    ax.axhline(0, color='#64748B', linewidth=1)
    ax.set_ylim(-0.25, 0.25)
    ax.set_ylabel('Reward Shape Impact', color='#0F172A', fontsize=9, fontweight='bold')
    ax.set_title('DeepScalper Hindsight Reward Shaping Architecture', color='#0F172A', fontsize=11, fontweight='bold', pad=12)

    for bar in bars:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        y_pos = height + 0.01 if height >= 0 else height - 0.02
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f"{height:+.2f}", ha='center', va=va, color='#0F172A', fontsize=9, fontweight='bold')

    ax.tick_params(colors='#1E293B', labelsize=8)
    ax.grid(True, axis='y', color='#E2E8F0', linestyle='--', alpha=0.7)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "retro_reward_shaping_chart.png")
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Generated retro_reward_shaping_chart.png")
    return path


if __name__ == "__main__":
    generate_candlestick_chart()
    generate_architecture_flowchart()
    generate_reward_shaping_chart()
