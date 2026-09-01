import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

arms = [
    {"label": "control\n(bootstrap)", "conf": 0.6808044005392675, "deconf": 0.6807442982014585,
     "delta": -6.01e-05, "p": 0.347,  "control": True},
    {"label": "control\n(weights)",   "conf": 0.6808044005392675, "deconf": 0.6808044005392675,
     "delta": 0.0,       "p": 1.000,  "control": True},
    {"label": "g-as[a]\n(bootstrap)", "conf": 0.6808044005392675, "deconf": 0.6633718828778016,
     "delta": -0.01743,  "p": 1.94e-14, "control": False},
    {"label": "ug-as[a]\n(bootstrap)","conf": 0.6808044005392675, "deconf": 0.6637664576444402,
     "delta": -0.01704,  "p": 5.41e-15, "control": False},
    {"label": "ug-as[a]\n(weights)",  "conf": 0.6808044005392675, "deconf": 0.6607636204095463,
     "delta": -0.02004,  "p": 8.53e-19, "control": False},
]

def p_label(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"

COLOR_CONF_MAIN   = "#4C72B0"
COLOR_DECONF_MAIN = "#C44E52"
COLOR_CONF_CTRL   = "#9AAABF"
COLOR_DECONF_CTRL = "#C8A8A8"

n     = len(arms)
x     = np.arange(n, dtype=float)
width = 0.32
off   = width / 2 + 0.04

fig, ax = plt.subplots(figsize=(11, 5))

for i, arm in enumerate(arms):
    cc = COLOR_CONF_CTRL   if arm["control"] else COLOR_CONF_MAIN
    dc = COLOR_DECONF_CTRL if arm["control"] else COLOR_DECONF_MAIN
    alpha = 0.6 if arm["control"] else 1.0

    b_conf   = ax.bar(x[i] - off, arm["conf"],   width=width,
                      color=cc, alpha=alpha, zorder=3)
    b_deconf = ax.bar(x[i] + off, arm["deconf"], width=width,
                      color=dc, alpha=alpha, zorder=3)

    for bar, val in [(b_conf, arm["conf"]), (b_deconf, arm["deconf"])]:
        ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                val - 0.005,
                f"{val:.4f}", ha="center", va="top",
                fontsize=7.5, color="white", fontweight="bold")

    y_top = max(arm["conf"], arm["deconf"]) + 0.006
    ax.annotate("", xy=(x[i] + off, y_top), xytext=(x[i] - off, y_top),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    label = f"Δ = {arm['delta']:+.4f}\n{p_label(arm['p'])}"
    ax.text(x[i], y_top + 0.006, label,
            ha="center", va="bottom", fontsize=8,
            color="black" if not arm["control"] else "#555555")

for i, arm in enumerate(arms):
    if arm["control"]:
        ax.axvspan(x[i] - 0.5, x[i] + 0.5, color="#f0f0f0", zorder=0)

ax.set_xticks(x)
ax.set_xticklabels([a["label"] for a in arms], fontsize=9.5)
ax.set_ylabel("Balanced Accuracy", fontsize=11)
ax.set_ylim(0.62, 0.73)
ax.set_xlim(-0.6, n - 0.4)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(handles=[
    mpatches.Patch(color=COLOR_CONF_MAIN,   label="Confounded"),
    mpatches.Patch(color=COLOR_DECONF_MAIN, label="De-confounded"),
    mpatches.Patch(color=COLOR_CONF_CTRL,   alpha=0.6, label="Control arms (shaded)"),
], fontsize=9.5, framealpha=0.9, loc="lower right")

ax.set_title(
    "NeuroVoz intra-corpus de-confounding: 200 repeated stratified splits, 10 bootstrap resamples",
    fontsize=10.5, pad=10
)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_nv_deconf.png")
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved → {out_path}")