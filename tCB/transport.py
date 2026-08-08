
import sys, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

BASE = "/Users/reecepalmer/Downloads/transportable causal bootstrapping"
DATA = f"{BASE}/data"
LBS  = f"{BASE}/libraries"
for p in [f"{LBS}/GRAPL", f"{LBS}/CausalBootstrapping", BASE]:
    if p not in sys.path:
        sys.path.insert(0, p)

N_FEATS = 66
INTENSITY_IDX = slice(0, 38)
PAUSE_IDX = slice(38, 59)
SYLLABLE_IDX = slice(59, 66)

def _stem(x):
    s = os.path.basename(str(x))
    while s.endswith(".wav"):
        s = s[:-4]
    return s.replace("audios-", "").replace("audios/", "")

def load_neurovoz():
    base = f"{DATA}/neurovoz/extracted_features"
    intens = pd.read_csv(f"{base}/intensity_features_neurovoz.csv")
    pause  = pd.read_csv(f"{base}/pause_features_neurovoz.csv")
    syll   = pd.read_csv(f"{base}/syllable_features_neurovoz.csv")
    meta = {"Unnamed: 0", "tot_name"}
    keys_i = intens["tot_name"].apply(_stem)
    keys_p = pause["tot_name"].apply(_stem)
    keys_s = syll["tot_name"].apply(_stem)
    common = sorted(set(keys_i) & set(keys_p) & set(keys_s))
    i_df = intens.loc[keys_i.isin(common)].loc[:,~intens.columns.isin(meta)].set_index(keys_i[keys_i.isin(common)].values)
    p_df = pause.loc[keys_p.isin(common)].loc[:,~pause.columns.isin(meta)].set_index(keys_p[keys_p.isin(common)].values)
    s_df = syll.loc[keys_s.isin(common)].loc[:,~syll.columns.isin(meta)].set_index(keys_s[keys_s.isin(common)].values)
    X = pd.concat([i_df, p_df, s_df], axis=1).values.astype(np.float64)
    Y = np.array([1 if s.startswith("PD_") else 0 for s in common], dtype=np.float64)
    return X, Y

def load_parkceleb():
    base = f"{DATA}/parkceleb_extracted"
    intens = pd.read_csv(f"{base}/intensity_features_parkceleb.csv")
    pause  = pd.read_csv(f"{base}/pause_features_parkceleb.csv")
    syll   = pd.read_csv(f"{base}/syllable_features_parkceleb.csv")
    meta = {"parkceleb_group","parkceleb_label","speaker_id","video_id",
            "diarization_speaker","source_type","audio_file","audio_ext","audio_path",
            "intensity_ok","intensity_error","pause_ok","pause_error",
            "syllable_ok","syllable_error","tot_name"}
    i_df = intens.loc[:,~intens.columns.isin(meta)]
    p_df = pause.loc[:,~pause.columns.isin(meta)]
    s_df = syll.loc[:,~syll.columns.isin(meta)]
    X = pd.concat([i_df, p_df, s_df], axis=1).values.astype(np.float64)
    Y = intens["parkceleb_label"].values.astype(np.float64)
    return X, Y

def run_sID():
    import sID_expr as sid_alg
    import grapl.dsl as dsl
    D = "U; X; Y; S; U -> X; U -> Y; X -> Y; S -> U;"
    g = dsl.GraplDSL().readgrapl(D)
    result = sid_alg.sid(Y={"Y"}, X={"X"}, S={"S"}, D=g)
    if result is not None:
        return str(result)
    return None

def estimate_confound(nv_X, pc_X, nv_Y, pc_Y, method="pca_min_corr"):
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

def compute_tcb_weights(U_src, U_trg, Y_trg):
    pu_src = gaussian_kde(U_src.ravel())(U_trg.ravel())
    pu_trg = gaussian_kde(U_trg.ravel())(U_trg.ravel())

    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=42)
    lr.fit(U_trg.reshape(-1, 1), Y_trg.ravel())
    py1 = lr.predict_proba(U_trg.reshape(-1, 1))[:, 1]
    py_given_u = np.where(Y_trg.ravel() > 0.5, py1, 1 - py1)

    w = pu_src / (pu_trg * py_given_u)
    w = w / np.mean(w)
    return w

def train_clf(train_X, train_Y, test_X, test_Y, sample_weight=None, seed=42):
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=seed)
    clf.fit(train_X, train_Y, sample_weight=sample_weight)
    p = clf.predict_proba(test_X)[:, 1]
    auc = roc_auc_score(test_Y, p)
    acc = accuracy_score(test_Y, (p > 0.5).astype(float))
    f1  = f1_score(test_Y, (p > 0.5).astype(float))
    return dict(accuracy=acc, auc_roc=auc, f1=f1)

def main():
    N_REPS = 10
    print(f"  sID transport formula: {run_sID()}\n")

    nv_X, nv_Y = load_neurovoz()
    pc_X, pc_Y = load_parkceleb()

    for domain, X, Y in [("NeuroVoz", nv_X, nv_Y), ("ParkCeleb", pc_X, pc_Y)]:
        ok = ~np.isnan(X).any(axis=1)
        if ok.sum() < len(ok):
            print(f"  Dropped {len(ok)-ok.sum()} NaN rows from {domain}")

    nv_X = nv_X[~np.isnan(nv_X).any(axis=1)]
    nv_Y = nv_Y[:len(nv_X)]
    pc_X = pc_X[~np.isnan(pc_X).any(axis=1)]
    pc_Y = pc_Y[:len(pc_X)]

    print(f"  NeuroVoz: {len(nv_X)} samples (PD={int(nv_Y.sum())}, HC={len(nv_Y)-int(nv_Y.sum())})")
    print(f"  ParkCeleb: {len(pc_X)} samples (PD={int(pc_Y.sum())}, HC={len(pc_Y)-int(pc_Y.sum())})")

    scaler = StandardScaler()
    pool_X = np.vstack([nv_X, pc_X])
    pool_X = scaler.fit_transform(pool_X)
    nv_X_s = pool_X[:len(nv_X)]
    pc_X_s = pool_X[len(nv_X):]

    U_nv, U_pc = estimate_confound(nv_X_s, pc_X_s, nv_Y, pc_Y, method="pca_min_corr")

    results = {k: [] for k in ["Standard", "TCB-Weighted"]}
    for rep in range(N_REPS):
        seed = 42 + rep
        rng = np.random.default_rng(seed)
        N_pc = len(pc_X_s)
        perm = rng.permutation(N_pc)
        n_train = int(0.8 * N_pc)
        i_tr, i_te = perm[:n_train], perm[n_train:]

        tr_X, tr_Y = pc_X_s[i_tr], pc_Y[i_tr]
        te_X, te_Y = pc_X_s[i_te], pc_Y[i_te]

        tcb_w = compute_tcb_weights(U_nv, U_pc[i_tr], tr_Y)

        results["Standard"].append(train_clf(tr_X, tr_Y, te_X, te_Y, seed=seed))
        results["TCB-Weighted"].append(train_clf(tr_X, tr_Y, te_X, te_Y,
                                                  sample_weight=tcb_w, seed=seed))

        print(f"  Rep {rep+1:2d}/{N_REPS}:  "
              f"Std AUC={results['Standard'][-1]['auc_roc']:.3f}  "
              f"TCB AUC={results['TCB-Weighted'][-1]['auc_roc']:.3f}")

    print(f"\n  {'─'*55}")
    print(f"  {'Aggregated over 10 reps':^55}")
    print(f"  {'─'*55}")
    print(f"  {'Method':<20} {'Acc':>8} {'AUC':>8} {'F1':>8}")
    print(f"  {'─'*55}")
    for key, label in [("Standard", "Standard (ParkCeleb)"),
                       ("TCB-Weighted", "TCB Transported")]:
        aucs = [r["auc_roc"] for r in results[key]]
        accs = [r["accuracy"] for r in results[key]]
        f1s  = [r["f1"] for r in results[key]]
        print(f"  {label:<20} {np.mean(accs):>6.3f}±{np.std(accs):.3f}  "
              f"{np.mean(aucs):>6.3f}±{np.std(aucs):.3f}  "
              f"{np.mean(f1s):>6.3f}±{np.std(f1s):.3f}")
    print(f"  {'─'*55}")

if __name__ == "__main__":
    main()
