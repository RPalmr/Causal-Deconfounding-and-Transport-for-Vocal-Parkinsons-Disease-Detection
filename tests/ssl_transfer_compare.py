"""self-supervised vs hand-crafted feature transfer comparison for the dissertation.

loads neurovoz petaca blanca clips and parkceleb full set with paired
wav2vec2 embeddings, then reports transfer accuracy/auc under raw and
causal-bootstrapping arms across multiple pca dimensionalities."""

import sys, os, glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

BASE = "/Volumes/ESSD/transportable causal bootstrapping"
for p in (f"{BASE}/tests", f"{BASE}/libraries/CausalBootstrapping",
          f"{BASE}/libraries/GRAPL", f"{BASE}/libraries/causal_assistant",
          f"{BASE}/tCB"):
    if p not in sys.path:
        sys.path.insert(0, p)
import transport_dissertation_replica as tdr

SSL_NV = f"{BASE}/data/ssl_nv"
SSL_PC = f"{BASE}/data/ssl_pc"

def nv_petaca_stems():
    hc = pd.read_csv(f"{BASE}/data/neurovoz/metadata/metadata_hc.csv")
    pd_ = pd.read_csv(f"{BASE}/data/neurovoz/metadata/metadata_pd.csv")
    mt = pd.concat([hc, pd_], ignore_index=True)
    mt["_pid"] = mt["ID"].map(lambda i: f"{int(i):04d}")
    mt["Age"] = pd.to_numeric(mt["Age"], errors="coerce")
    a_by_pid = mt.groupby("_pid")["Age"].median()
    intens = pd.read_csv(f"{BASE}/data/neurovoz/extracted_features/"
                         f"intensity_features_neurovoz.csv")
    pause = pd.read_csv(f"{BASE}/data/neurovoz/extracted_features/"
                        f"pause_features_neurovoz.csv")
    syll = pd.read_csv(f"{BASE}/data/neurovoz/extracted_features/"
                       f"syllable_features_neurovoz.csv")
    ki = intens["tot_name"].apply(tdr._stem)
    kp = pause["tot_name"].apply(tdr._stem)
    ks = syll["tot_name"].apply(tdr._stem)
    common = sorted(set(ki) & set(kp) & set(ks))
    common = [c for c in common if "PETACA" in c]
    X = pd.concat([_sl(intens, ki, common, tdr.META_COLS),
                   _sl(pause, kp, common, tdr.META_COLS),
                   _sl(syll, ks, common, tdr.META_COLS)], axis=1)
    X = X.values.astype(np.float64)
    pid = np.array([c[-4:] for c in common])
    ag = np.array([a_by_pid.get(p, np.nan) for p in pid], dtype=np.float64)
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(ag))
    return [c for c, o in zip(common, ok) if o]

def _sl(df, keys, common, drop):
    return df.set_index(keys).loc[common].drop(columns=drop, errors="ignore")

def main():

    X66, ys, ags, pids = tdr.load_nv()
    stems = nv_petaca_stems()
    assert len(stems) == len(X66), (len(stems), len(X66))

    have = [c for c in stems if os.path.exists(f"{SSL_NV}/{c}.npy")]
    idx = [stems.index(c) for c in have]
    E = np.array([np.load(f"{SSL_NV}/{c}.npy") for c in have])
    yN = ys[idx]; agN = ags[idx]
    XN = X66[idx]
    print(f"NV source: {len(have)} clips with SSL (PD={int(yN.sum())})")

    mu, sd = E.mean(0), E.std(0)
    sd = np.maximum(sd, np.median(sd) * 0.05)
    EN = (E - mu) / sd

    m66 = X66.mean(0)
    s66 = np.maximum(X66.std(0), np.median(X66.std(0)) * 0.05)
    Xn66 = (X66 - m66) / s66

    pc = tdr.load_pc()
    Xt66, yt, sidt, agt = pc["full"]
    intens = pd.read_csv(f"{BASE}/data/parkceleb_extracted/"
                         f"intensity_features_parkceleb.csv")
    kid = intens["audio_file"].astype(str).map(
        lambda f: (lambda m: m.group(1) if m else np.nan)(tdr.age.IDRE.search(f)))
    m_row = kid.notna().values
    assert Xt66.shape[0] == int(m_row.sum()), (Xt66.shape[0], m_row.sum())
    af = intens.loc[m_row, "audio_file"].astype(str) \
        .str.replace(".wav", "", regex=False).values
    have_t = np.array([os.path.exists(f"{SSL_PC}/{s}.npy") for s in af])
    n_t = int(have_t.sum())
    if n_t == 0:
        print("no PC SSL embeddings yet"); return
    X66t = Xt66[have_t]
    yt2 = yt[have_t]
    ET = np.array([np.load(f"{SSL_PC}/{s}.npy")
                   for s in af[have_t]])
    ET = (ET - mu) / sd
    X66ts = (X66t - m66) / s66
    print(f"PC Full target: {n_t} clips with SSL (y PD {int(yt2.sum())})")

    yN_i = yN.astype(int)

    W = tdr._cb(Xn66, ys, ags)
    w_cb = W[idx, ys[idx].astype(int)]

    def fit(Xtr):
        out = {}
        for arm, w in [("raw", None), ("cb", w_cb)]:
            clf = XGBClassifier(n_estimators=100, learning_rate=0.1,
                                max_depth=3, min_child_weight=3,
                                random_state=42, verbosity=0)
            if w is not None:
                w = w * (len(yN) / w.sum())
            clf.fit(Xtr, yN_i, sample_weight=w)
            out[arm] = clf
        return out

    print(f"\nNV → PC-Full transfer (identical {n_t} clip subset)")
    print(f"{'space':<8}{'arm':<5}{'acc':>6}{'BA':>7}{'F1':>7}{'AUC':>7}")
    for space, XN_f, XT_f in (("SSL", EN, ET), ("66-dim", Xn66, X66ts)):
        for arm, clf in fit(XN_f).items():
            pr = clf.predict_proba(XT_f)[:, 1]
            m = tdr._metrics(yt2, pr)
            print(f"{space:<8}{arm:<5}{m[0]:>6.3f}{m[1]:>7.3f}"
                  f"{m[2]:>7.3f}{m[3]:>7.3f}")

    def auc(Xtr, Xte, use_cb_):
        clf = XGBClassifier(n_estimators=100, learning_rate=0.1,
                            max_depth=3, min_child_weight=3,
                            random_state=42, verbosity=0)
        w = None
        if use_cb_:
            w = w_cb * (len(yN) / w_cb.sum())
        clf.fit(Xtr, yN_i, sample_weight=w)
        return tdr._metrics(yt2, clf.predict_proba(Xte)[:, 1])[3]

    print(f"\nSSL full-dim AUC: raw {auc(EN, ET, False):.3f}, "
          f"cb {auc(EN, ET, True):.3f}")
    for k in (5, 10, 20, 40, 80, 160):
        mu2, comps = tdr._pca_fit(EN, k)
        ENk = tdr._pca_apply(EN, (mu2, comps))
        ETk = tdr._pca_apply(ET, (mu2, comps))
        print(f"SSL PCA{k:>3}: raw {auc(ENk, ETk, False):.3f}  "
              f"cb {auc(ENk, ETk, True):.3f}")

    print("\ncontrol — 66-dim PCA sweep:")
    for k in (5, 10, 20, 40):
        mu2, comps = tdr._pca_fit(Xn66, min(k, Xn66.shape[1]))
        Xk = tdr._pca_apply(Xn66, (mu2, comps))
        Tk = tdr._pca_apply(X66ts, (mu2, comps))
        print(f"66 PCA{k:>3}: raw {auc(Xk, Tk, False):.3f}  "
              f"cb {auc(Xk, Tk, True):.3f}")

    print("\nSSL PCA5 source-CV (strata):")
    for tr, te in tdr._strat_folds(yN.astype(int), 3, 42):
        mu2, comps = tdr._pca_fit(EN[tr], 5)
        clf = XGBClassifier(n_estimators=100, learning_rate=0.1,
                            max_depth=3, min_child_weight=3,
                            random_state=42, verbosity=0)
        clf.fit(tdr._pca_apply(EN[tr], (mu2, comps)), yN[tr].astype(int))
        mm = tdr._metrics(yN[te],
                          clf.predict_proba(tdr._pca_apply(EN[te],
                                                           (mu2, comps)))[:, 1])
        print(f"  fold: BA {mm[1]:.3f} AUC {mm[3]:.3f}")

if __name__ == "__main__":
    main()