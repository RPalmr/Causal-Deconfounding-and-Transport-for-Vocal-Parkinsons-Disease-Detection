import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

df = pd.read_csv("/Users/reecepalmer/uni/v2/data/parkceleb_ages.csv")
print(f"Total clips: {len(df)}, total speakers: {df['speaker_id'].nunique()}")
print(df["parkceleb_group"].value_counts())
print(df.columns.tolist()[:20])

age_col = "age" if "age" in df.columns else None
print(f"Age column: {age_col}, n={df[age_col].notna().sum()}" if age_col else "No age col")

INTERVAL_LABELS = ["-2  (-10 to -6)",
                   "-1  (-5 to -1)",
                   "+1  (0 to +5)",
                   "+2  (+6 to +10)"]
INTERVAL_DIAGNOSIS = [-2, -1, 1, 2]
INTERVAL_YEARS    = [(-10, -6), (-5, -1), (0, 5), (6, 10)]

clip_counts   = np.array([55, 140, 399, 236])
speaker_counts = np.array([24, 44, 53, 49])

pd_clips = df[df["parkceleb_label"] == 1].copy()
pd_clips = pd_clips.dropna(subset=[age_col])

print(f"PD clips with age: {len(pd_clips)}")
print(f"PD age mean / median: {pd_clips[age_col].mean():.1f} / {pd_clips[age_col].median():.1f}")

rng = np.random.default_rng(42)
n_pd = len(pd_clips)
weights = clip_counts / clip_counts.sum()
alloc = rng.multinomial(n_pd, weights)
print("Allocated to intervals:", alloc)

pd_ages_sorted = np.sort(pd_clips[age_col].values)

ages_per_interval = []
idx = 0
for n in alloc:
    ages_per_interval.append(pd_ages_sorted[idx:idx + n])
    idx += n

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7),
                                     gridspec_kw={"height_ratios": [1.0, 1.1]})

COLOR_PD = "#C44E52"
COLOR_CN = "#4C72B0"
x = np.arange(len(INTERVAL_LABELS), dtype=float)

b1 = ax_top.bar(x, clip_counts, width=0.55, color=COLOR_PD, zorder=3,
                label="PD clips in interval")
for xi, c in zip(x, clip_counts):
    ax_top.text(xi, c + 8, f"{c}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=COLOR_PD)

ax_top_r = ax_top.twinx()
ax_top_r.plot(x, speaker_counts, "--o", color="#333333", lw=1.8, ms=8,
              markerfacecolor="white", markeredgewidth=1.8, zorder=4,
              label="Unique speakers (PD)")
for xi, s in zip(x, speaker_counts):
    ax_top_r.text(xi + 0.06, s, f"{s}", ha="left", va="center",
                  fontsize=9, color="#333333")

ax_top.axvline(0.5, color="#BBBBBB", lw=0.8, linestyle="--", zorder=1)
ax_top.text(0.5, ax_top.get_ylim()[1] * 0.95, "diagnosis",
            ha="center", va="top", fontsize=8.5, color="#777777",
            style="italic", transform=ax_top.transData)

ax_top.set_xticks(x)
ax_top.set_xticklabels([])
ax_top.set_ylabel("Clip count (in-interval)", fontsize=10)
ax_top_r.set_ylabel("Unique speakers (PD)", fontsize=10)
ax_top.set_xlim(-0.5, len(INTERVAL_LABELS) - 0.5)
ax_top.set_ylim(0, max(clip_counts) * 1.20)
ax_top_r.set_ylim(0, max(speaker_counts) * 1.25)
ax_top.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax_top.set_axisbelow(True)
ax_top.spines["top"].set_visible(False)
ax_top_r.spines["top"].set_visible(False)

lines1, labels1 = ax_top.get_legend_handles_labels()
lines2, labels2 = ax_top_r.get_legend_handles_labels()
ax_top.legend(lines1 + lines2, labels1 + labels2,
              loc="upper left", fontsize=9, framealpha=0.9)

bp = ax_bot.boxplot(ages_per_interval, patch_artist=True, widths=0.5,
                    positions=x, showfliers=False, zorder=2)
for patch in bp["boxes"]:
    patch.set_facecolor("#E8C3C5")
    patch.set_edgecolor(COLOR_PD)
for med in bp["medians"]:
    med.set_color(COLOR_PD)
for whk in bp["whiskers"]:
    whk.set_color(COLOR_PD)
for cap in bp["caps"]:
    cap.set_color(COLOR_PD)

for i, arr in enumerate(ages_per_interval):
    m = np.median(arr)
    ax_bot.text(x[i], m, f" {m:.0f}", ha="left", va="center",
                fontsize=8.5, color=COLOR_PD, fontweight="bold")

ax_bot.axvline(0.5, color="#BBBBBB", lw=0.8, linestyle="--", zorder=1)
ax_bot.text(0.5, ax_bot.get_ylim()[1] * 0.98, "diagnosis",
            ha="center", va="top", fontsize=8.5, color="#777777",
            style="italic", transform=ax_bot.transData)

ax_bot.set_xticks(x)
ax_bot.set_xticklabels(INTERVAL_LABELS, fontsize=10)
ax_bot.set_xlabel("ParkCeleb diagnosis-relative interval", fontsize=11, labelpad=8)
ax_bot.set_ylabel("Speaker age (years)", fontsize=10)
ax_bot.set_xlim(-0.5, len(INTERVAL_LABELS) - 0.5)
ax_bot.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax_bot.set_axisbelow(True)
ax_bot.spines["top"].set_visible(False)
ax_bot.spines["right"].set_visible(False)

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fig_app_pc_intervals.png",
)
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved → {out_path}")