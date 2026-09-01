import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

intervals = ["-2\n(-10 to -6 yrs)", "-1\n(-5 to -1 yrs)",
             "+1\n(0 to +5 yrs)",   "+2\n(+6 to +10 yrs)"]
x = np.array([0, 1, 2, 3], dtype=float)

conf   = [0.534965034965035,    0.6590909090909091,
          0.5890313390313391,   0.43561872909698995]
deconf = [0.513986013986014,    0.47727272727272724,
          0.6225071225071226,   0.43143812709030105]
deltas = [-0.02098, -0.18182, +0.03348, -0.00418]
p_vals = [1.53e-24, 1.55e-114, 1.32e-27, 3.60e-09]

fig, ax = plt.subplots(figsize=(10, 5.5))

COLOR_CONF   = "#4C72B0"
COLOR_DECONF = "#C44E52"

ax.plot(x, conf,   "-o", color=COLOR_CONF,   lw=2.0, ms=9,
        label="Confounded",    zorder=4)
ax.plot(x, deconf, "-s", color=COLOR_DECONF, lw=2.0, ms=9,
        label="De-confounded (UGAS)", zorder=4)

for xi, c, d in zip(x, conf, deconf):
    ax.plot([xi, xi], [c, d], color="#888888", lw=1.0,
            linestyle=":", zorder=2)

for xi, c, d in zip(x, conf, deconf):
    ax.text(xi - 0.05, c + 0.012, f"{c:.3f}",
            ha="right", va="bottom", fontsize=8.5, color=COLOR_CONF)
    ax.text(xi + 0.05, d - 0.018, f"{d:.3f}",
            ha="left",  va="top",   fontsize=8.5, color=COLOR_DECONF)

for xi, c, d, dv, pv in zip(x, conf, deconf, deltas, p_vals):
    y_mid = (c + d) / 2
    sign  = "+" if dv >= 0 else ""
    txt   = f"Δ = {sign}{dv:.3f}\np < 0.001"
    if dv < 0:
        ax.text(xi + 0.13, y_mid + 0.025, txt,
                ha="left", va="center", fontsize=8,
                color="black",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="#FFFDF2", ec="#888888", lw=0.6))
    else:
        ax.text(xi + 0.13, y_mid - 0.025, txt,
                ha="left", va="center", fontsize=8,
                color="black",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="#EAF7E0", ec="#888888", lw=0.6))

ax.set_xticks(x)
ax.set_xticklabels(intervals, fontsize=10)
ax.set_xlabel("ParkCeleb diagnosis-relative interval", fontsize=11, labelpad=8)
ax.set_ylabel("Balanced Accuracy\n(NeuroVoz Petaca → ParkCeleb)", fontsize=11)
ax.set_xlim(-0.4, 3.6)
ax.set_ylim(0.35, 0.74)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.axvline(0.5, color="#BBBBBB", lw=0.8, linestyle="--", zorder=1)
ax.text(0.5, 0.715, "diagnosis", ha="center", va="bottom",
        fontsize=8.5, color="#777777", style="italic")

ax.legend(fontsize=10, framealpha=0.9, loc="lower left")

ax.set_title(
    "UGAS cross-corpus transport: 200 repetitions, 10 bootstrap resamples",
    fontsize=11, pad=10
)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig5_ugas_transport.png")
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved → {out_path}")