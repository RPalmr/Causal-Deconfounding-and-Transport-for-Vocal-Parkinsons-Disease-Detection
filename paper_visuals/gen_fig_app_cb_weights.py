import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, "/Users/reecepalmer/uni/v2")
from main import nv_data, nv_weights
from old.causal import GRAPH_MODELS  # noqa: E402

X, y, ag, sx, pid, up, gr = nv_data()
y = np.asarray(y).astype(int)
ag = np.asarray(ag)
sx = np.asarray(sx)

print(f"NV: n={len(y)}, PD={y.sum()}, HC={(y==0).sum()}")
print(f"PD mean age: {ag[y==1].mean():.2f}, HC mean age: {ag[y==0].mean():.2f}")

graph = GRAPH_MODELS["g-as[a]"]
W = nv_weights(graph, X, y, ag, sx, up, gr)
print(f"Weights shape: {W.shape}, sum PD: {W[y==1, 1].sum():.2f}, "
      f"sum HC: {W[y==0, 0].sum():.2f}")

fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 4.2),
                                          sharey=True)

COLOR_PD = "#C44E52"
COLOR_HC = "#4C72B0"

bins = np.arange(30, 100, 5)

ax_before.hist(ag[y == 1], bins=bins, density=True, alpha=0.55,
               color=COLOR_PD, label=f"PD (n={int((y==1).sum())}, mean={ag[y==1].mean():.1f})")
ax_before.hist(ag[y == 0], bins=bins, density=True, alpha=0.55,
               color=COLOR_HC, label=f"HC (n={int((y==0).sum())}, mean={ag[y==0].mean():.1f})")
ax_before.axvline(ag[y == 1].mean(), color=COLOR_PD, ls="--", lw=1.3)
ax_before.axvline(ag[y == 0].mean(), color=COLOR_HC, ls="--", lw=1.3)
ax_before.set_xlabel("Age (years)", fontsize=11)
ax_before.set_ylabel("Density", fontsize=11)
ax_before.set_title("Before reweighting (marginal)", fontsize=11)
ax_before.legend(fontsize=9, framealpha=0.9)
ax_before.spines["top"].set_visible(False)
ax_before.spines["right"].set_visible(False)

w = W[np.arange(len(y)), y]

w_pd = W[y == 1, 1]
w_hc = W[y == 0, 0]

def weighted_hist(ax, ages, weights, color, label):
    norm = weights.sum()
    if norm <= 0:
        return
    ax.hist(ages, bins=bins, weights=weights / norm,
            density=False, alpha=0.55, color=color, label=label)
    ax.set_ylabel("Density", fontsize=11)
    ax.clear()
    ax.hist(ages, bins=bins, weights=weights,
            density=True, alpha=0.55, color=color, label=label)

def weighted_kde(ages, weights, grid):
    ages = np.asarray(ages, float)
    weights = np.asarray(weights, float)
    if weights.sum() <= 0:
        return np.zeros_like(grid)
    h = 1.06 * ages.std() * len(ages) ** (-1/5) if len(ages) > 1 else 1.0
    h = max(h, 1.0)
    diff = (grid[:, None] - ages[None, :]) / h
    kde = (weights[None, :] * np.exp(-0.5 * diff ** 2)).sum(axis=1)
    kde /= (weights.sum() * h * np.sqrt(2 * np.pi))
    return kde

def plot_weighted_hist(ax, ages, weights, color, label, bins, grid):
    weights = np.asarray(weights, float)
    if weights.sum() <= 0:
        return
    ax.hist(ages, bins=bins, weights=weights,
            density=True, alpha=0.55, color=color, label=label)
    kde = weighted_kde(ages, weights, grid)
    ax.plot(grid, kde, color=color, lw=1.8)
    wmean = np.average(ages, weights=weights)
    ax.axvline(wmean, color=color, ls="--", lw=1.3)
    return wmean

grid = np.linspace(30, 100, 400)
pd_mean_after = plot_weighted_hist(
    ax_after, ag[y == 1], w_pd, COLOR_PD,
    f"PD (n={int((y==1).sum())}, w-mean={np.average(ag[y==1], weights=w_pd):.1f})",
    bins, grid,
)
hc_mean_after = plot_weighted_hist(
    ax_after, ag[y == 0], w_hc, COLOR_HC,
    f"HC (n={int((y==0).sum())}, w-mean={np.average(ag[y==0], weights=w_hc):.1f})",
    bins, grid,
)
ax_after.set_xlabel("Age (years)", fontsize=11)
ax_after.set_title("After reweighting (Age+Sex back-door)", fontsize=11)
ax_after.legend(fontsize=9, framealpha=0.9)
ax_after.spines["top"].set_visible(False)
ax_after.spines["right"].set_visible(False)

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fig_app_cb_weights.png",
)
fig.tight_layout()
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved → {out_path}")