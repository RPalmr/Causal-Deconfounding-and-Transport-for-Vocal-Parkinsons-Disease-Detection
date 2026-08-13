import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as _SS
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import causal as c

X_raw, ys, ags, sxs, _, ups, grs, tasks = c.load_nv()
X_raw_pb, ys_pb, ags_pb, sxs_pb, _, ups_pb, grs_pb = c.load_nv(task="PETACA_BLANCA")
mu_s, sd_s = X_raw.mean(0), X_raw.std(0)
sd_s = np.maximum(sd_s, np.median(sd_s) * 0.05)
Xs = (X_raw - mu_s) / sd_s
Xs_pb = (X_raw_pb - mu_s) / sd_s
pc = c.load_pc()

pca = PCA(n_components=5).fit(Xs[:, c.INTENSITY_IDX])
L_nv = pca.transform(Xs[:, c.INTENSITY_IDX])
corrs = np.array([abs(np.corrcoef(L_nv[:, i], ys)[0, 1]) for i in range(5)])
k5 = np.argsort(corrs)
print("pc corrs:", np.round(corrs, 3), "order:", k5 + 1)

def run_window(X_pc, y_pc, sid_pc, age_pc, k, clip, reps=200):
    m0 = ~np.isnan(X_pc).any(axis=1)
    X_pc, y_pc, sid_pc, age_pc = X_pc[m0], y_pc[m0], sid_pc[m0], age_pc[m0]
    if len(np.unique(sid_pc)) < 4:
        return None
    U_nv = L_nv[:, k]
    res = {"standard": [], "localised": [], "tcb": [], "tcbc": []}
    for rep in range(reps):
        tr, te = c._speaker_split(sid_pc, y_pc, c.RANDOM_SEED + rep)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        sc = _SS().fit(X_pc[tr])
        X_tr, X_te = sc.transform(X_pc[tr]), sc.transform(X_pc[te])
        y_tr, y_te = y_pc[tr], y_pc[te]
        U_tr = pca.transform(X_tr[:, c.INTENSITY_IDX])[:, k]
        w_loc = c._loc_weights(X_tr, y_tr, age_pc[tr])
        w_loc = w_loc[np.arange(len(y_tr)), y_tr.astype(int)]
        w_tcb = c._tcb_weights(U_nv, U_tr, y_tr)
        w_tcbc = c._tcb_weights(U_nv, U_tr, y_tr, clip)
        g_std = c._gbc(X_tr, y_tr)
        g_loc = c._gbc(X_tr, y_tr, w_loc)
        g_tcb = c._gbc(X_tr, y_tr, w_tcb)
        g_tcbc = c._gbc(X_tr, y_tr, w_tcbc)
        res["standard"].append(c._metrics(y_te, g_std.predict_proba(X_te)[:, 1]))
        res["localised"].append(c._metrics(y_te, g_loc.predict_proba(X_te)[:, 1]))
        res["tcb"].append(c._metrics(y_te, g_tcb.predict_proba(X_te)[:, 1]))
        res["tcbc"].append(c._metrics(y_te, g_tcbc.predict_proba(X_te)[:, 1]))
    return res

def ba_line(res, name):
    a = np.asarray(res[name])
    ref = np.asarray(res["standard"])
    d = a[:, 1] - ref[:, 1]
    d = d[np.isfinite(d)]
    p = c._perm_gt(a[:, 1], ref[:, 1])
    s = np.nanmean(a[:, 1])
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    return f"{name:10s} ba={s:.4f} p={p:.4f} [{lo:.4f}, {hi:.4f}]"

print("\n--- u dim audit (ba, vs standard) ---")
for d in (1, 2, 3):
    k = k5[:d]
    print(f"u dims={d} pcs={k + 1}")
    for name in ("full", "-2", "-1", "1", "2"):
        X_pc, y_pc, sid_pc, age_pc = pc[name]
        clip = (1.0, 99.0)
        res = run_window(X_pc, y_pc, sid_pc, age_pc, k, clip)
        if res is None:
            print("  " + name + ": skipped")
            continue
        print("  " + name + ":")
        for arm in ("standard", "localised", "tcb", "tcbc"):
            a = np.asarray(res[arm])
            print("    " + f"{arm:10s} acc={np.nanmean(a[:, 0]):.4f} ba={np.nanmean(a[:, 1]):.4f}" + ("" if arm == "standard" else f"  p={c._perm_gt(a[:, 1], np.asarray(res['standard'])[:, 1]):.4f}"))

print("\n--- tcb clip levels at canonical u (dims=2) ---")
k = k5[:2]
for name in ("full", "-2", "-1", "1", "2"):
    X_pc, y_pc, sid_pc, age_pc = pc[name]
    line = name + ":"
    for clip in (None, (5.0, 95.0), (1.0, 99.0), (0.5, 99.5)):
        res = run_window(X_pc, y_pc, sid_pc, age_pc, k, clip)
        a = np.asarray(res["tcbc"] if clip is not None else res["tcb"])
        p = c._perm_gt(a[:, 1], np.asarray(res["standard"])[:, 1])
        line += f"  clip={clip}: ba={np.nanmean(a[:, 1]):.4f} p={p:.4f}"
    print(line)

print("\n--- fine-tune sweep (model=both, windows full/-2/-1) ---")
W_base = c._cb(Xs, ys, ags, sxs, model="both", up=ups, gr=grs)
w_base = W_base[np.arange(len(ys)), ys.astype(int)]
W_spec = c._cb(Xs_pb, ys_pb, ags_pb, sxs_pb, model="both", up=ups_pb, gr=grs_pb)
w_spec = W_spec[np.arange(len(ys_pb)), ys_pb.astype(int)]
wins = {}
for name in ("full", "-2", "-1"):
    m0 = ~np.isnan(pc[name][0]).any(axis=1)
    wins[name] = ((pc[name][0][m0] - mu_s) / sd_s, pc[name][1][m0])
for et in (10, 20, 40):
    for flr in (0.1, 0.05, 0.01):
        if et == 20 and flr == 0.1:
            continue
        g_cb = c._gbc_two_stage(Xs, ys, w_base, Xs_pb, ys_pb, w_spec, extra_trees=et, fine_lr=flr)
        g_raw = c._gbc_two_stage(Xs, ys, None, Xs_pb, ys_pb, None, extra_trees=et, fine_lr=flr)
        out = f"et={et} lr={flr}:"
        for name, (Xt, yt) in wins.items():
            mc = np.asarray(c._metrics(yt, g_cb.predict_proba(Xt)[:, 1]))
            mr = np.asarray(c._metrics(yt, g_raw.predict_proba(Xt)[:, 1]))
            out += f"  {name}: cb ba={mc[1]:.4f} f1={mc[2]:.4f} | raw ba={mr[1]:.4f} f1={mr[2]:.4f}"
        print(out)