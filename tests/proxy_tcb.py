"""
Proxy-Mediated Transportable Causal Bootstrapping
==================================================
Audio-predicted clinical proxies (GRBAS / UPDRS / tremor / age) as U.

Stage 1: Train clinical regressors on NeuroVoz audio features → clinical measure
Stage 2: Apply to ParkCeleb audio → proxy U_pc
Stage 3: TCB reweights ParkCeleb toward NeuroVoz's clinical distribution
         w = P_nv(U) / P_pc(U, Y)

The proxy is meaningful only if it predicts the label (r(U, Y) > 0) and
shifts across domains.
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ -> root
DATA = f"{BASE}/data"
for p in [f"{BASE}/libraries/GRAPL", f"{BASE}/libraries/CausalBootstrapping", BASE]:
    if p not in sys.path:
        sys.path.insert(0, p)

N_FEATS = 66

def _stem(x):
    s = os.path.basename(str(x))
    while s.endswith(".wav"):
        s = s[:-4]
    return s.replace("audios-", "").replace("audios/", "")

def _name(a):
    return os.path.splitext(os.path.basename(str(a)))[0]

def _load_neurovoz_meta():
    """Features + clinical metadata + GRBAS ratings, all keyed by recording."""
    base = f"{DATA}/neurovoz/extracted_features"
    intens = pd.read_csv(f"{base}/intensity_features_neurovoz.csv")
    pause  = pd.read_csv(f"{base}/pause_features_neurovoz.csv")
    syll   = pd.read_csv(f"{base}/syllable_features_neurovoz.csv")
    meta_cols = {"Unnamed: 0", "tot_name"}
    ki = intens["tot_name"].apply(_stem)
    kp = pause["tot_name"].apply(_stem)
    ks = syll["tot_name"].apply(_stem)
    common = sorted(set(ki) & set(kp) & set(ks))
    def sl(df, kv):
        return df.loc[kv.isin(common)].loc[:,~df.columns.isin(meta_cols)].set_index(kv[kv.isin(common)].values)
    X = pd.concat([sl(intens, ki), sl(pause, kp), sl(syll, ks)], axis=1).values.astype(np.float64)
    Y = np.array([1 if s.startswith("PD_") else 0 for s in common], dtype=np.float64)

    hc = pd.read_csv(f"{DATA}/neurovoz/metadata/metadata_hc.csv")
    pd_ = pd.read_csv(f"{DATA}/neurovoz/metadata/metadata_pd.csv")
    mt = pd.concat([hc, pd_], ignore_index=True)
    mt["_k"] = mt["Audio"].apply(_name)
    idx = mt.set_index("_k")

    def _mget(col, default=np.nan):
        return np.array([idx.loc[k, col] if k in idx.index else default for k in common], dtype=np.float64)

    age    = _mget("Age")
    tremor = np.where(_mget("Vocal tremor") == 1, 1, 0)
    hypoph = np.where(_mget("Hypophonic voice") == 1, 1, 0)
    updrs  = np.where(Y == 0, 0.0, _mget("UPDRS scale"))

    # GRBAS: keyed by task_id (e.g. A1_0034); NV keys carry PD_/HC_ prefix
    grbas = {}
    for f in glob.glob(f"{DATA}/grbas/*.csv"):
        gdf = pd.read_csv(f)
        for _, r in gdf.iterrows():
            gid = os.path.splitext(os.path.basename(str(r["text_patient_id"])))[0].strip()
            try:
                grbas[gid] = float(r["G"])
            except (ValueError, TypeError):
                pass
    gscore = np.array([grbas.get(k[3:], np.nan) for k in common], dtype=np.float64)

    ok = ~(np.isnan(X).any(axis=1) | np.isnan(age) | np.isnan(tremor) | np.isnan(hypoph)
           | np.isnan(updrs) | np.isnan(gscore))
    return (X[ok], Y[ok], age[ok], tremor[ok], hypoph[ok], updrs[ok], gscore[ok])


def _load_parkceleb():
    base = f"{DATA}/parkceleb_extracted"
    intens = pd.read_csv(f"{base}/intensity_features_parkceleb.csv")
    pause  = pd.read_csv(f"{base}/pause_features_parkceleb.csv")
    syll   = pd.read_csv(f"{base}/syllable_features_parkceleb.csv")
    meta = {"parkceleb_group","parkceleb_label","speaker_id","video_id",
            "diarization_speaker","source_type","audio_file","audio_ext","audio_path",
            "intensity_ok","intensity_error","pause_ok","pause_error",
            "syllable_ok","syllable_error","tot_name"}
    X = pd.concat([
        intens.loc[:,~intens.columns.isin(meta)],
        pause.loc[:,~pause.columns.isin(meta)],
        syll.loc[:,~syll.columns.isin(meta)],
    ], axis=1).values.astype(np.float64)
    Y = intens["parkceleb_label"].values.astype(np.float64)
    ok = ~np.isnan(X).any(axis=1)
    return X[ok], Y[ok]


def tcb_weights(U_src, U_trg, Y_trg):
    """w = P_src(U) / P_trg(U, Y), normalised to mean=1."""
    pu_src = gaussian_kde(U_src.ravel())(U_trg.ravel())
    pu_trg = gaussian_kde(U_trg.ravel())(U_trg.ravel())
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    lr.fit(U_trg.reshape(-1, 1), Y_trg.ravel())
    py1 = lr.predict_proba(U_trg.reshape(-1, 1))[:, 1]
    pgu = np.where(Y_trg.ravel() > 0.5, py1, 1 - py1)
    w = pu_src / (pu_trg * pgu)
    w /= np.mean(w)
    return w


def train_clf(train_X, train_Y, test_X, test_Y, sample_weight=None, seed=42):
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=seed)
    clf.fit(train_X, train_Y, sample_weight=sample_weight)
    p = clf.predict_proba(test_X)[:, 1]
    return dict(auc=roc_auc_score(test_Y, p),
                acc=accuracy_score(test_Y, (p > 0.5).astype(float)),
                f1=f1_score(test_Y, (p > 0.5).astype(float)))


def main():
    N_REPS = 100
    print("  Proxy-mediated TCB (NV clinical → PC):")
    t0 = time.time()

    nv_X, nv_Y, nv_age, nv_tremor, nv_hp, nv_updrs, nv_grbas = _load_neurovoz_meta()
    pc_X, pc_Y = _load_parkceleb()
    print(f"    NeuroVoz (with clinical+GRBAS): {len(nv_Y)} samples")
    print(f"    ParkCeleb:                      {len(pc_Y)} samples")

    # Pooled standardisation
    scaler = StandardScaler()
    pool = np.vstack([nv_X, pc_X])
    pool = scaler.fit_transform(pool)
    nv_X = pool[:len(nv_Y)]
    pc_X = pool[len(nv_Y):]

    # NV train/test split (for proxy quality check)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(nv_Y))
    n_tr = int(0.8 * len(nv_Y))
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]

    # Stage 1: train proxy regressors on NV
    targets = {
        "age":    (nv_age, False),
        "tremor": (nv_tremor,  True),
        "hypoph": (nv_hp,  True),
        "updrs":  (nv_updrs, False),
        "grbasG": (nv_grbas, False),
    }
    proxies = {}
    for name, (y, is_binary) in targets.items():
        if is_binary:
            m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
            m.fit(nv_X[tr_idx], y[tr_idx])
            p_tr = m.predict_proba(nv_X[tr_idx])[:, 1]
            p_te = m.predict_proba(nv_X[te_idx])[:, 1]
            p_pc = m.predict_proba(pc_X)[:, 1]
            q_tr = roc_auc_score(y[tr_idx], p_tr)
            q_te = roc_auc_score(y[te_idx], p_te)
        else:
            m = LinearRegression()
            m.fit(nv_X[tr_idx], y[tr_idx])
            p_tr = m.predict(nv_X[tr_idx])
            p_te = m.predict(nv_X[te_idx])
            p_pc = m.predict(pc_X)
            q_tr = np.corrcoef(y[tr_idx], p_tr)[0, 1]
            q_te = np.corrcoef(y[te_idx], p_te)[0, 1]
        proxies[name] = (np.concatenate([p_tr, p_te]), p_pc)  # (U_nv_all, U_pc)
        print(f"      {name:<8} proxy  train q={q_tr:+.3f}  held-out q={q_te:+.3f}")

    # Stage 3: TCB cross-domain with each proxy as U
    print()
    results = {}
    for name, (U_nv, U_pc) in proxies.items():
        r_pc = abs(np.corrcoef(U_pc, pc_Y)[0, 1])
        results[name] = {"std": [], "tcb": [], "r_pc": r_pc}
        for rep in range(N_REPS):
            seed = 42 + rep
            r = np.random.default_rng(seed)
            perm_pc = r.permutation(len(pc_Y))
            n_pc_tr = int(0.8 * len(pc_Y))
            pc_tr, pc_te = perm_pc[:n_pc_tr], perm_pc[n_pc_tr:]
            tr_X, tr_Y = pc_X[pc_tr], pc_Y[pc_tr]
            te_X, te_Y = pc_X[pc_te], pc_Y[pc_te]

            w = tcb_weights(U_nv, U_pc[pc_tr], tr_Y)
            results[name]["std"].append(train_clf(tr_X, tr_Y, te_X, te_Y, seed=seed)["auc"])
            results[name]["tcb"].append(train_clf(tr_X, tr_Y, te_X, te_Y,
                                                  sample_weight=w, seed=seed)["auc"])

    print(f"    {'proxy U':<10} {'r(U,Y)_PC':>9} {'Std AUC':>8} {'TCB AUC':>8} {'gain':>8}")
    for name, res in results.items():
        s = np.mean(res["std"]); t = np.mean(res["tcb"])
        print(f"    {name:<10} {res['r_pc']:>9.3f} {s:>8.3f} {t:>8.3f} {t-s:>+8.4f}")

    print(f"\n  Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
