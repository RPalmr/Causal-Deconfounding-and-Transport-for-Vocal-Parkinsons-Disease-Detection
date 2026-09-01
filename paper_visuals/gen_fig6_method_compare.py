import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

intervals = ["-2  (-10 to -6 yrs)",
             "-1  (-5 to -1 yrs)",
             "+1  (0 to +5 yrs)",
             "+2  (+6 to +10 yrs)"]

conf_baseline = {
    "-2": 0.534965034965035,
    "-1": 0.6590909090909091,
    "+1": 0.5890313390313391,
    "+2": 0.43561872909698995,
}

ugas_deconf = {
    "-2": 0.513986013986014,
    "-1": 0.47727272727272724,
    "+1": 0.6225071225071226,
    "+2": 0.43143812709030105,
}
ugas_delta = {
    "-2": -0.02098, "-1": -0.18182, "+1": +0.03348, "+2": -0.00418,
}

sid_deconf = {
    "-2": 0.534965034965035,
    "-1": 0.5454545454545454,
    "+1": 0.6396011396011396,
    "+2": 0.5275919732441472,
}
sid_delta = {
    "-2":  0.0,
    "-1": -0.11364,
    "+1": +0.05057,
    "+2": +0.09197,
}

fig, ax = plt.subplots(figsize=(11, 5.5))

y        = np.arange(len(intervals), dtype=float)
bar_h    = 0.32
y_ugas   = y + bar_h / 2 + 0.02
y_sid    = y - bar_h / 2 - 0.02

COLOR_UGAS = "#4C72B0"
COLOR_SID  = "#C44E52"
COLOR_CONF = "#666666"

for i, label in enumerate(intervals):
    key = label[:2]
    ax.barh(y_ugas[i], ugas_deconf[key], height=bar_h,
            color=COLOR_UGAS, zorder=3,
            label="UGAS" if i == 0 else None)
    ax.barh(y_sid[i],  sid_deconf[key],  height=bar_h,
            color=COLOR_SID,  zorder=3,
            label="sID-proxy" if i == 0 else None)

    ax.text(ugas_deconf[key] + 0.005, y_ugas[i], f"{ugas_deconf[key]:.3f}",
            ha="left", va="center", fontsize=8, color=COLOR_UGAS)
    ax.text(sid_deconf[key]  + 0.005, y_sid[i],  f"{sid_deconf[key]:.3f}",
            ha="left", va="center", fontsize=8, color=COLOR_SID)

    cb = conf_baseline[key]
    ax.plot([cb, cb], [y_sid[i] - bar_h / 2, y_ugas[i] + bar_h / 2],
            color=COLOR_CONF, lw=1.6, linestyle="--", zorder=4)
    ax.text(cb, y_ugas[i] + bar_h / 2 + 0.12, f"conf = {cb:.3f}",
            ha="center", va="bottom", fontsize=7.5, color=COLOR_CONF)

    sign_u = "+" if ugas_delta[key] >= 0 else ""
    sign_s = "+" if sid_delta[key]  >= 0 else ""
    txt_u = f"Δ = {sign_u}{ugas_delta[key]:.3f}"
    txt_s = f"Δ = {sign_s}{sid_delta[key]:.3f}"
    x_max = max(ugas_deconf[key], sid_deconf[key])
    ax.text(x_max + 0.045, y_ugas[i], txt_u,
            ha="left", va="center", fontsize=8, color=COLOR_UGAS)
    ax.text(x_max + 0.045, y_sid[i],  txt_s,
            ha="left", va="center", fontsize=8, color=COLOR_SID)

ax.set_yticks(y)
ax.set_yticklabels(intervals, fontsize=10)
ax.set_xlabel("Balanced Accuracy", fontsize=11)
ax.set_xlim(0.35, 0.85)
ax.invert_yaxis()
ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.axhline(0.5, color="#BBBBBB", lw=0.8, linestyle="-", zorder=1)
ax.text(0.355, 0.5, "diagnosis", ha="left", va="center",
        fontsize=8.5, color="#777777", style="italic")

legend_handles = [
    mpatches.Patch(color=COLOR_UGAS, label="UGAS de-confounded"),
    mpatches.Patch(color=COLOR_SID,  label="sID-proxy de-confounded"),
    plt.Line2D([0], [0], color=COLOR_CONF, lw=1.6, linestyle="--",
               label="Confounded baseline (shared)"),
]
ax.legend(handles=legend_handles, fontsize=10, framealpha=0.9,
          loc="lower right")

ax.set_title(
    "UGAS vs sID-proxy cross-corpus transport: ΔBA per ParkCeleb interval",
    fontsize=11, pad=10
)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig6_method_compare.png")
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved → {out_path}")