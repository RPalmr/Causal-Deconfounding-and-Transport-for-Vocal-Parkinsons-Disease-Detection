import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import old.causal as c

OUT = str(c.BASE) + "/outputs"
RANDOM_SEED = c.RANDOM_SEED

XFER_COLS = ["interval", "conf", "deconf", "dacc", "p_ks", "reps", "cb_repeats"]


def done(path, key_cols, vals):
    try:
        d = pd.read_csv(path, dtype=str)
        return (pd.Index(key_cols).isin(d.columns).all()
                and tuple(map(str, vals)) in set(zip(*[d[k].values for k in key_cols])))
    except Exception:
        return False


def save(path, key_cols, cols, rows):
    df = pd.DataFrame(rows, columns=cols)
    if os.path.exists(path):
        try:
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        except Exception:
            pass
    for k in key_cols:
        df[k] = df[k].astype(str)
    df.drop_duplicates(subset=key_cols, keep="last").to_csv(path, index=False)


def ks_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return 1.0
    return float(ks_2samp(a, b, alternative="two-sided").pvalue)


def speaker_ba(sid, y, score):
    y = np.asarray(y).astype(int)
    sid = np.asarray(sid)
    score = np.asarray(score)
    yg, pg = [], []
    for s in np.unique(sid):
        m = sid == s
        yg.append(np.bincount(y[m]).argmax())
        pg.append(score[m].mean())
    return c._balanced_acc(np.array(yg), (np.array(pg) >= 0.5).astype(int))


def pca_confounder_weights(U_nv, nv_Y, U_pc, y_t):
    try:
        pu_src = gaussian_kde(U_nv.ravel())(U_nv.ravel())
    except Exception:
        pu_src = np.ones_like(U_nv.ravel(), dtype=float)
    try:
        pu_trg = gaussian_kde(U_pc.ravel())(U_nv.ravel())
    except Exception:
        pu_trg = np.ones_like(U_nv.ravel(), dtype=float)
    pu_src = np.where(np.isfinite(pu_src) & (pu_src > 0), pu_src, 1e-12)
    pu_trg = np.where(np.isfinite(pu_trg) & (pu_trg > 0), pu_trg, 1e-12)

    lr = LogisticRegression(C=1e6, solver="liblinear", max_iter=10000,
                            random_state=RANDOM_SEED)
    lr.fit(U_pc.reshape(-1, 1), y_t.astype(int))
    py1 = lr.predict_proba(U_nv.reshape(-1, 1))[:, 1]
    py_given_u = np.where(nv_Y > 0, py1, 1 - py1)
    py_given_u = np.where((py_given_u > 0) & np.isfinite(py_given_u),
                          py_given_u, 1e-12)

    W = pu_src / (pu_trg * py_given_u)
    W = W / max(np.mean(W), 1e-12)
    lo, hi = np.percentile(W, 1), np.percentile(W, 99)
    W = np.clip(W, lo, hi)
    return W


def transfer(xfer_reps, cb_repeats):
    Xp, yp, _, _, _, _, _ = c.load_nv(task="PETACA_BLANCA")
    nv_X = np.asarray(Xp, dtype=float)
    nv_Y = np.asarray(yp).astype(int)
    nv_X = nv_X[~np.isnan(nv_X).any(axis=1)]
    nv_Y = nv_Y[:len(nv_X)]

    nv_int = nv_X[:, :38] if nv_X.shape[1] >= 38 else nv_X
    scaler = StandardScaler().fit(nv_int)
    nv_int_s = scaler.transform(nv_int)

    pca = PCA(n_components=min(5, nv_int_s.shape[1]))
    L_nv = pca.fit_transform(np.nan_to_num(nv_int_s, nan=0.0))
    corrs = np.array([abs(np.corrcoef(L_nv[:, i], nv_Y)[0, 1])
                      for i in range(L_nv.shape[1])])
    best_k = int(np.argmin(corrs))
    U_nv = L_nv[:, best_k]

    intervals = c.load_pc()
    path = OUT + "/pipeline_transfer_pca.csv"
    rows = []
    for name, payload in tqdm(intervals.items(), desc="transfer intervals"):
        if done(path, ["interval", "reps", "cb_repeats"],
                [name, xfer_reps, cb_repeats]):
            continue
        if len(payload) == 4:
            X_t, y_t, sid_t, _ = payload
        else:
            X_t, y_t, sid_t = payload[0], payload[1], payload[2]
        m0 = ~np.isnan(X_t).any(axis=1)
        X_t, y_t, sid_t = X_t[m0], y_t[m0], sid_t[m0]
        if len(y_t) < 4 or len(np.unique(y_t)) < 2:
            continue

        pc_int = X_t[:, :38] if X_t.shape[1] >= 38 else X_t
        pc_int_s = scaler.transform(np.nan_to_num(pc_int, nan=0.0))
        U_pc = pca.transform(np.nan_to_num(pc_int_s, nan=0.0))[:, best_k]

        W = pca_confounder_weights(U_nv, nv_Y, U_pc, y_t)

        med = {"conf": [], "dec": []}
        for rep in tqdm(range(xfer_reps), desc=f"  interval {name}", leave=False):
            seed = RANDOM_SEED + rep
            gd = c._gbc(Xp, yp, W, seed=seed)
            gc = c._gbc(Xp, yp, seed=seed)
            med["dec"].append(speaker_ba(sid_t, y_t,
                                         gd.predict_proba(X_t)[:, 1]))
            med["conf"].append(speaker_ba(sid_t, y_t,
                                          gc.predict_proba(X_t)[:, 1]))
        mc, md = float(np.median(med["conf"])), float(np.median(med["dec"]))
        p = ks_p(med["dec"], med["conf"])
        rows.append([name, mc, md, md - mc, p, xfer_reps, cb_repeats])
    if rows:
        save(path, ["interval"], XFER_COLS, rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xfer-reps", type=int, default=200)
    ap.add_argument("--cb-repeats", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()
    if args.out_dir is not None:
        OUT = args.out_dir
    os.makedirs(OUT, exist_ok=True)
    transfer(args.xfer_reps, args.cb_repeats)