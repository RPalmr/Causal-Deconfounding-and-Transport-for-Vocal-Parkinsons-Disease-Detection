import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

data = {
    "conf->conf": 0.8861109029142346,
    "conf->rev":  0.7345127638479474,
    "dec->conf":  0.8721874232430792,
    "dec->rev":   0.8199924547663925,
}

conf_matched  = data["conf->conf"]
conf_reversed = data["conf->rev"]
dec_matched   = data["dec->conf"]
dec_reversed  = data["dec->rev"]

delta_conf = conf_reversed - conf_matched
delta_dec  = dec_reversed  - dec_matched

fig, ax = plt.subplots(figsize=(7, 4.5))

x      = np.array([0.0, 1.0])
width  = 0.32
offset = width / 2 + 0.04

color_matched  = "#4C72B0"
color_reversed = "#C44E52"

bars_matched  = ax.bar(x - offset, [conf_matched,  dec_matched],
                       width=width, color=color_matched,  zorder=3,
                       label="Matched test set")
bars_reversed = ax.bar(x + offset, [conf_reversed, dec_reversed],
                       width=width, color=color_reversed, zorder=3,
                       label="Reversed test set")

def annotate_delta(ax, x_left, x_right, y_top, delta, pad=0.012):
    y = y_top + pad
    ax.annotate(
        "", xy=(x_right, y), xytext=(x_left, y),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
    )
    ax.text((x_left + x_right) / 2, y + 0.008,
            f"Δ = {delta:+.3f}",
            ha="center", va="bottom", fontsize=9, color="black")

top_conf = max(conf_matched, conf_reversed)
top_dec  = max(dec_matched,  dec_reversed)

annotate_delta(ax, x[0] - offset, x[0] + offset, top_conf, delta_conf)
annotate_delta(ax, x[1] - offset, x[1] + offset, top_dec,  delta_dec)

for bar in list(bars_matched) + list(bars_reversed):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h - 0.025,
            f"{h:.3f}", ha="center", va="top",
            fontsize=8.5, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["Confounded\nclassifier", "De-confounded\nclassifier"],
                   fontsize=11)
ax.set_ylabel("Balanced Accuracy", fontsize=11)
ax.set_ylim(0.65, 0.98)
ax.set_xlim(-0.55, 1.55)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(handles=[
    mpatches.Patch(color=color_matched,  label="Matched test set"),
    mpatches.Patch(color=color_reversed, label="Reversed test set"),
], fontsize=10, framealpha=0.9, loc="lower right")

ax.set_title(
    "Semi-synthetic validation: matched vs. reversed sex confounding",
    fontsize=11, pad=10
)

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "fig1_semisynth.png")
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved → {out_path}")