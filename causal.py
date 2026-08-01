"""
Transportable Causal Bootstrapping: NeuroVoz → ParkCeleb
========================================================
Bareinboim & Pearl (2013) transportability framework applied to
Parkinson's detection with 66-dim handcrafted acoustic features.

Selection diagram:  U → X, U → Y, X → Y, S → U
  - U: latent confound (recording condition, age, clinical factors)
  - X: 66-dim acoustic features
  - Y: PD/HC label
  - S: domain selection (NeuroVoz vs ParkCeleb)

TCB reweights the target samples so P_trg(U) → P_src(U),
removing the domain-specific confound bias.
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = f"{BASE}/data"
for p in [f"{BASE}/libraries/GRAPL", f"{BASE}/libraries/CausalBootstrapping", BASE]:
    if p not in sys.path:
        sys.path.insert(0, p)

N_FEATS = 66
INTENSITY_IDX = slice(0, 38)
PAUSE_IDX = slice(38, 59)
SYLLABLE_IDX = slice(59, 66)


"""data loaders here"""

def _stem(x):
    s = os.path.basename(str(x))
    while s.endswith(".wav"):
        s = s[:-4]
    return s.replace("audios-", "").replace("audios/", "")

def _name(a):
    return os.path.splitext(os.path.basename(str(a)))[0]

def load_neurovoz(include_metadata=True):
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

    if not include_metadata:
        ok = ~np.isnan(X).any(axis=1)
        return X[ok], Y[ok]

    hc = pd.read_csv(f"{DATA}/neurovoz/metadata/metadata_hc.csv")
    pd_ = pd.read_csv(f"{DATA}/neurovoz/metadata/metadata_pd.csv")
    mt = pd.concat([hc, pd_], ignore_index=True)
    mt["_k"] = mt["Audio"].apply(_name)
    idx = mt.set_index("_k")

    def _mget(col, default=np.nan):
        return np.array([idx.loc[k, col] if k in idx.index else default for k in common], dtype=np.float64)

    age    = _mget("Age")
    sex    = _mget("Sex")
    tremor = np.where(_mget("Vocal tremor") == 1, 1, 0)
    hypoph = np.where(_mget("Hypophonic voice") == 1, 1, 0)
    updrs  = np.where(Y == 0, 0.0, _mget("UPDRS scale"))

    ok = ~(np.isnan(X).any(axis=1) | np.isnan(age) | np.isnan(sex) | np.isnan(tremor) | np.isnan(hypoph) | np.isnan(updrs))
    return X[ok], Y[ok], age[ok], sex[ok], tremor[ok], hypoph[ok], updrs[ok]


def load_parkceleb():
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


"""sID transport formula"""

def run_sID():
    import sID_expr as sid_alg
    import grapl.dsl as dsl
    D = "U; X; Y; S; U -> X; U -> Y; X -> Y; S -> U;"
    g = dsl.GraplDSL().readgrapl(D)
    result = sid_alg.sid(Y={"Y"}, X={"X"}, S={"S"}, D=g)
    return result.tostr() if result is not None else "NOT TRANSPORTABLE"


"""tcb importance weights"""

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


"""confound estimation"""

def estimate_confound(nv_X, pc_X, nv_Y, pc_Y, method="pca_min_corr"):
    """Estimate latent confound U from the domain shift between NeuroVoz and ParkCeleb."""
    if method == "pca_min_corr":
        pca = PCA(n_components=5)
        L_nv = pca.fit_transform(nv_X[:, INTENSITY_IDX])
        corrs = np.array([abs(np.corrcoef(L_nv[:, i], nv_Y)[0, 1])
                          for i in range(5)])
        best_k = int(np.argmin(corrs))
        U_nv = L_nv[:, best_k]
        U_pc = pca.transform(pc_X[:, INTENSITY_IDX])[:, best_k]
        print(f"    U = PC{best_k+1}  (|r(Y)|={corrs[best_k]:.3f})")
    elif method == "pca_src":
        pca = PCA(n_components=1)
        pca.fit(nv_X[:, INTENSITY_IDX])
        U_nv = pca.transform(nv_X[:, INTENSITY_IDX]).ravel()
        U_pc = pca.transform(pc_X[:, INTENSITY_IDX]).ravel()
    elif method == "mean_diff":
        mu_nv = np.mean(nv_X[:, INTENSITY_IDX], axis=0)
        mu_pc = np.mean(pc_X[:, INTENSITY_IDX], axis=0)
        d = mu_pc - mu_nv
        d = d / np.linalg.norm(d)
        U_nv = nv_X[:, INTENSITY_IDX] @ d
        U_pc = pc_X[:, INTENSITY_IDX] @ d
    return U_nv, U_pc


"""training helper"""

def train_clf(train_X, train_Y, test_X, test_Y, sample_weight=None, seed=42):
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=seed)
    clf.fit(train_X, train_Y, sample_weight=sample_weight)
    p = clf.predict_proba(test_X)[:, 1]
    return dict(auc=roc_auc_score(test_Y, p),
                acc=accuracy_score(test_Y, (p > 0.5).astype(float)),
                f1=f1_score(test_Y, (p > 0.5).astype(float)))


"""Cross-domain (NeuroVoz → ParkCeleb)"""

def cross_domain():
    """
    Transport causal knowledge from NeuroVoz (source) to ParkCeleb (target).
    
    TCB reweights ParkCeleb samples to match NeuroVoz's confound distribution,
    removing the domain-specific bias.
    """
    N_REPS = 1000
    print("  Cross-domain transport NeuroVoz → ParkCeleb:")

    nv_X, nv_Y = load_neurovoz(include_metadata=False)
    pc_X, pc_Y = load_parkceleb()

    nv_ok = ~np.isnan(nv_X).any(axis=1)
    pc_ok = ~np.isnan(pc_X).any(axis=1)
    nv_X, nv_Y = nv_X[nv_ok], nv_Y[nv_ok]
    pc_X, pc_Y = pc_X[pc_ok], pc_Y[pc_ok]

    print(f"    NeuroVoz: {len(nv_Y)} samples (PD={int(nv_Y.sum())}, HC={len(nv_Y)-int(nv_Y.sum())})")
    print(f"    ParkCeleb: {len(pc_Y)} samples (PD={int(pc_Y.sum())}, HC={len(pc_Y)-int(pc_Y.sum())})")

    scaler = StandardScaler()
    pool_X = np.vstack([nv_X, pc_X])
    pool_X = scaler.fit_transform(pool_X)
    nv_X = pool_X[:len(nv_Y)]
    pc_X = pool_X[len(nv_Y):]

    U_nv, U_pc = estimate_confound(nv_X, pc_X, nv_Y, pc_Y, method="pca_min_corr")

    results = {k: [] for k in ["Standard", "TCB-Weighted"]}
    for rep in range(N_REPS):
        seed = 42 + rep
        rng = np.random.default_rng(seed)
        N_pc = len(pc_X)
        perm = rng.permutation(N_pc)
        n_train = int(0.8 * N_pc)
        i_tr, i_te = perm[:n_train], perm[n_train:]

        tr_X, tr_Y = pc_X[i_tr], pc_Y[i_tr]
        te_X, te_Y = pc_X[i_te], pc_Y[i_te]

        tcb_w = tcb_weights(U_nv, U_pc[i_tr], tr_Y)

        results["Standard"].append(train_clf(tr_X, tr_Y, te_X, te_Y, seed=seed))
        results["TCB-Weighted"].append(train_clf(tr_X, tr_Y, te_X, te_Y,
                                                   sample_weight=tcb_w, seed=seed))

        print(f"    Rep {rep+1:2d}/{N_REPS}:  "
              f"Std AUC={results['Standard'][-1]['auc']:.3f}  "
              f"TCB AUC={results['TCB-Weighted'][-1]['auc']:.3f}")

    auc_std = np.mean([r["auc"] for r in results["Standard"]])
    auc_tcb = np.mean([r["auc"] for r in results["TCB-Weighted"]])
    print(f"\n    Standard (ParkCeleb):       AUC={auc_std:.3f}")
    print(f"    TCB Transported (NV→PC):    AUC={auc_tcb:.3f}")
    print(f"    Causal transport gain:       {auc_tcb - auc_std:+.4f}")
    return results


"""core code structure"""

if __name__ == "__main__":
    t0 = time.time()
    print(f"  sID transport formula: {run_sID()}\n")
    cross_domain()
    print(f"\n  Time: {time.time()-t0:.0f}s")
