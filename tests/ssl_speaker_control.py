import sys, os
import numpy as np, pandas as pd
from xgboost import XGBClassifier

BASE = "/Volumes/ESSD/transportable causal bootstrapping"
for p in (f"{BASE}/tests", f"{BASE}/libraries/CausalBootstrapping",
          f"{BASE}/libraries/GRAPL", f"{BASE}/libraries/causal_assistant",
          f"{BASE}/tCB"):
    if p not in sys.path:
        sys.path.insert(0, p)
import transport_dissertation_replica as tdr
sys.path.insert(0, f"{BASE}/tests")
import ssl_transfer_compare as sc

def main():

    X66, yN, ags, pids = tdr.load_nv()
    m66 = X66.mean(0)
    s66 = np.maximum(X66.std(0), np.median(X66.std(0)) * 0.05)
    Xn = (X66 - m66) / s66

    stems = sc.nv_petaca_stems()
    have = [c for c in stems if os.path.exists(f"{BASE}/data/ssl_nv/{c}.npy")]
    idx = [stems.index(c) for c in have]
    Es = np.array([np.load(f"{BASE}/data/ssl_nv/{c}.npy") for c in have])
    muE = Es.mean(0)
    sdE = np.maximum(Es.std(0), np.median(Es.std(0)) * 0.05)
    EN = (Es - muE) / sdE
    yNs = yN[idx]
    print(f"source: {len(yNs)} clips")

    pc = tdr.load_pc()
    Xt66, yt, sidt0, agt = pc["full"]
    intens = pd.read_csv(f"{BASE}/data/parkceleb_extracted/"
                         f"intensity_features_parkceleb.csv")
    kid = intens["audio_file"].astype(str).map(
        lambda a: (lambda m: m.group(1) if m else np.nan)(
            tdr.age.IDRE.search(str(a))))
    m_row = kid.notnull().values
    af = intens.loc[m_row, "audio_file"].astype(str) \
        .str.replace(".wav", "", regex=False).values
    have_t = np.array([os.path.exists(f"{BASE}/data/ssl_pc/{s}.npy")
                       for s in af])
    X66t = (Xt66[have_t] - m66) / s66
    ET = np.array([np.load(f"{BASE}/data/ssl_pc/{s}.npy")
                   for s in af[have_t]])
    ET = (ET - muE) / sdE
    yt2 = yt[have_t].astype(int)
    sid2 = sidt0[have_t]
    print(f"target: {len(yt2)} clips, {len(np.unique(sid2))} speakers")
    sp_t = np.unique(sid2)

    def speaker_auc(prob):
        pv = np.array([prob[sid2 == s].mean() for s in sp_t])
        lv = np.array([yt2[sid2 == s][0] for s in sp_t])
        return tdr._roc_auc(lv.astype(int), pv)

    def fitm(Xsrc, ysrc, Xtgt, k=None, seed=42):
        if k:
            mu2, comps = tdr._pca_fit(Xsrc, k)
            Xsrc = tdr._pca_apply(Xsrc, (mu2, comps))
            Xtgt = tdr._pca_apply(Xtgt, (mu2, comps))
        cl = XGBClassifier(n_estimators=200, learning_rate=0.05,
                           max_depth=3, random_state=seed, verbosity=0)
        cl.fit(Xsrc, ysrc.astype(int))
        return speaker_auc(cl.predict_proba(Xtgt)[:, 1])

    print(f"\n{'space':<10}{'dims':<6}{'spkr-AUC raw':>14}{'cb':>7}")
    for label, Xs_, Xt_, k in (("66-dim", Xn, X66t, None),
                               ("66PCA", Xn, X66t, 5),
                               ("SSL", EN, ET, None),
                               ("SSL PCA5", EN, ET, 5)):
        a_raw = fitm(Xs_, yN, Xt_, k)
        W = tdr._cb(Xn, yN, ags)
        wc = W[np.arange(len(yN)), yN.astype(int)] * (len(yN) / W.sum())
        mu2c, comps_c = (None, None)
        Xs_c = Xs_
        if k:
            mu2c, comps_c = tdr._pca_fit(Xs_, k)
            Xs_c = tdr._pca_apply(Xs_, (mu2c, comps_c))
        cl = XGBClassifier(n_estimators=200, learning_rate=0.05,
                           max_depth=3, random_state=42, verbosity=0)
        cl.fit(Xs_c, yN.astype(int), sample_weight=wc)
        prob = cl.predict_proba(Xtgt_c if False else Xt_) [:, 1] if False else \
            cl.predict_proba(tdr._pca_apply(Xt_, (mu2c, comps_c))
                             if k else Xt_)[:, 1]
        a_cb = speaker_auc(prob)
        print(f"{label:<10}{str(k if k else 'all'):<6}{a_raw:>14.3f}{a_cb:>7.3f}")
    noise = np.random.RandomState(0).randn(len(Xn), 66)
    noise = (noise - noise.mean(0)) / np.maximum(noise.std(0), 1e-9)
    print("noise PCA  :", round(fitm(noise, yN, X66t, 5), 3))

if __name__ == "__main__":
    main()