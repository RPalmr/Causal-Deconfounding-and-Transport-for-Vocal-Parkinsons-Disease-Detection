import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

arms = [
    {"label": "age",         "n": 1506, "conf": 0.6307452534980624,
     "deconf": 0.6416003823853975, "delta": +0.01086, "p": 0.907},
    {"label": "accent",      "n": 1534, "conf": 0.632581212260185,
     "deconf": 0.638464718944171,  "delta": +0.00588, "p": 0.931},
    {"label": "quality",     "n": 814,  "conf": 0.6318181566779496,
     "deconf": 0.6241812792770385, "delta": -0.00764, "p": 0.414},
    {"label": "quality-dis", "n": 814,  "conf": 0.6318181566779496,
     "deconf": 0.6366393442622951, "delta": +0.00482, "p": 0.600},
]

def p_label(p):
    return f"p = {p:.3f}"

COLOR_CONF   = "#4C72B0"
COLOR_DECONF = "#C44E52"

n     = len(arms)
x     = np.arange(n, dtype=float)
width = 0.32
off   = width / 2 + 0.04

fig, ax = plt.subplots(figsize=(10, 5))

for i, arm in enumerate(arms):
    b_conf   = ax.bar(x[i] - off, arm["conf"],   width=width,
                      color=COLOR_CONF,   zorder=3,
                      label="Confounded" if i == 0 else None)
    b_deconf = ax.bar(x[i] + off, arm["deconf"], width=width,
                      color=COLOR_DECONF, zorder=3,
                      label="De-confounded" if i == 0 else None)

    for bar, val in [(b_conf, arm["conf"]), (b_deconf, arm["deconf"])]:
        ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                val - 0.003,
                f"{val:.3f}", ha="center", va="top",
                fontsize=8, color="white", fontweight="bold")

    ax.text(x[i], 0.605, f"n = {arm['n']}",
            ha="center", va="bottom", fontsize=8.5, color="#555555")

    y_top = max(arm["conf"], arm["deconf"]) + 0.006
    ax.annotate("", xy=(x[i] + off, y_top), xytext=(x[i] - off, y_top),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    label = f"Δ = {arm['delta']:+.4f}\n{p_label(arm['p'])}"
    ax.text(x[i], y_top + 0.005, label,
            ha="center", va="bottom", fontsize=8.5, color="black")

ax.set_xticks(x)
ax.set_xticklabels([a["label"] for a in arms], fontsize=11)
ax.set_ylabel("Balanced Accuracy", fontsize=11)
ax.set_ylim(0.60, 0.68)
ax.set_xlim(-0.5, n - 0.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(handles=[
    mpatches.Patch(color=COLOR_CONF,   label="Confounded"),
    mpatches.Patch(color=COLOR_DECONF, label="De-confounded"),
], fontsize=10, framealpha=0.9, loc="upper right")

ax.set_title(
    "ParkCeleb intra-corpus de-confounding: 50 repeated stratified splits, 10 bootstrap resamples\n"
    "no arm reaches statistical significance (p > 0.41)",
    fontsize=10, pad=10
)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig3_pc_deconf.png")
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved → {out_path}")