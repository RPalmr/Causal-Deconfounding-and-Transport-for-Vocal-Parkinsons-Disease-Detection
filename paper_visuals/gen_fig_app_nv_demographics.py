import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

pd_meta  = pd.read_csv("/Users/reecepalmer/uni/v2/data/neurovoz/metadata/metadata_pd.csv")
hc_meta  = pd.read_csv("/Users/reecepalmer/uni/v2/data/neurovoz/metadata/metadata_hc.csv")

pd_speakers = pd_meta.groupby("ID").first().reset_index()
hc_speakers = hc_meta.groupby("ID").first().reset_index()

excluded = {56, 74, 121, 135}
pd_speakers = pd_speakers[~pd_speakers["ID"].isin(excluded)]
hc_speakers = hc_speakers[~hc_speakers["ID"].isin(excluded)]

print("PD sex counts:", pd_speakers["Sex"].value_counts().to_dict())
print("HC sex counts:", hc_speakers["Sex"].value_counts().to_dict())
print(f"PD n={len(pd_speakers)}, HC n={len(hc_speakers)}")

fig, (ax_age, ax_sex) = plt.subplots(1, 2, figsize=(12, 4.2))

COLOR_PD = "#C44E52"
COLOR_HC = "#4C72B0"

pd_age = pd_speakers["Age"].dropna()
hc_age = hc_speakers["Age"].dropna()

bins = np.arange(30, 100, 5)
ax_age.hist(pd_age, bins=bins, density=True, alpha=0.55,
            color=COLOR_PD, label=f"PD (n={len(pd_age)}, mean={pd_age.mean():.1f})")
ax_age.hist(hc_age, bins=bins, density=True, alpha=0.55,
            color=COLOR_HC, label=f"HC (n={len(hc_age)}, mean={hc_age.mean():.1f})")

from scipy.stats import gaussian_kde
xs = np.linspace(30, 100, 300)
for arr, c in [(pd_age, COLOR_PD), (hc_age, COLOR_HC)]:
    kde = gaussian_kde(arr)
    ax_age.plot(xs, kde(xs), color=c, lw=1.8)

ax_age.axvline(pd_age.mean(), color=COLOR_PD, ls="--", lw=1.4)
ax_age.axvline(hc_age.mean(), color=COLOR_HC, ls="--", lw=1.4)
ax_age.text(pd_age.mean() + 0.5, ax_age.get_ylim()[1] * 0.92,
            f"{pd_age.mean():.1f}", color=COLOR_PD, fontsize=9)
ax_age.text(hc_age.mean() - 4.0, ax_age.get_ylim()[1] * 0.92,
            f"{hc_age.mean():.1f}", color=COLOR_HC, fontsize=9, ha="right")

ax_age.set_xlabel("Age (years)", fontsize=11)
ax_age.set_ylabel("Density", fontsize=11)
ax_age.set_xlim(30, 100)
ax_age.legend(fontsize=9, framealpha=0.9)
ax_age.spines["top"].set_visible(False)
ax_age.spines["right"].set_visible(False)
ax_age.set_title("Age distribution by diagnosis", fontsize=11)

pd_sex_counts = pd_speakers["Sex"].fillna(-1).value_counts()
hc_sex_counts = hc_speakers["Sex"].fillna(-1).value_counts()

def sex_label(v):
    return "Male" if v == 1.0 else ("Female" if v == 0.0 else "Unknown")

labels = ["Female", "Male"]
pd_fracs = [pd_sex_counts.get(0.0, 0), pd_sex_counts.get(1.0, 0)]
hc_fracs = [hc_sex_counts.get(0.0, 0), hc_sex_counts.get(1.0, 0)]
pd_total = sum(pd_fracs)
hc_total = sum(hc_fracs)
pd_pct = [f / pd_total * 100 for f in pd_fracs]
hc_pct = [f / hc_total * 100 for f in hc_fracs]

x = np.array([0, 1])
width = 0.35
b1 = ax_sex.bar(x - width/2, pd_pct, width, color=COLOR_PD,
                label=f"PD (n={pd_total})", zorder=3)
b2 = ax_sex.bar(x + width/2, hc_pct, width, color=COLOR_HC,
                label=f"HC (n={hc_total})", zorder=3)

for bars, pcts, counts in [(b1, pd_pct, pd_fracs), (b2, hc_pct, hc_fracs)]:
    for bar, pct, ct in zip(bars, pcts, counts):
        ax_sex.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                    f"{pct:.0f}%\n(n={ct})", ha="center", va="bottom",
                    fontsize=9, color="black")

ax_sex.set_xticks(x)
ax_sex.set_xticklabels(labels, fontsize=11)
ax_sex.set_ylabel("Percentage", fontsize=11)
ax_sex.set_ylim(0, 80)
ax_sex.legend(fontsize=9, framealpha=0.9, loc="upper right")
ax_sex.spines["top"].set_visible(False)
ax_sex.spines["right"].set_visible(False)
ax_sex.set_title("Sex composition by diagnosis", fontsize=11)

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fig_app_nv_demographics.png",
)
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved → {out_path}")