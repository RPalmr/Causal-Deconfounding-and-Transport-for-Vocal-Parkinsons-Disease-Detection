"""
Test: does TCB with a known confounder U recover the true causal relation
=========================================================================
Question:  Given a confounder U between PD status Y and voice features X,
           does TCB reweighting w = P_unconf(U) / (N*P_conf(U,Y)) recover
           the true causal relation between Y and X — i.e., is U sufficient
           to deconfound?

Design (text.txt / MNIST protocol, self-contained):
  - real NeuroVoz features X (causal signal: PD -> voice)
  - synthetic binary U:  Y ~ P(Y|U) with quy=0.85 (confounded) / 0.5 (causal)
  - X = X_real[Z] + U*brightness, Z ~ P(Z|Y) with qyz  (Pa(X) = {Z, U})
  - speaker-level split; TCB weights exact for binary U

Tests (all inside tests/, no code outside this folder):
  T1  weight identity : reweighted P_conf(U,Y) == P_unconf(U,Y)
  T2  recovery        : TCB-resampled (X,Y) indistinguishable from
                        unconfounded (X,Y) -- two-sample LR AUC ~ 0.5
  T3  sufficiency     : on the causal test set, U adds no label info
                        beyond X (AUC(Y|X,U) ~= AUC(Y|X))
  T4  positivity      : effective sample size of the TCB weights
  T5  mechanism       : recovery scales with qyz (causal path fidelity)
"""

import sys, os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ -> root
DATA = f"{BASE}/data"
N_FEATS = 66
QUY_CONF = 0.85
QYZ = 0.95
BRIGHT = 2.0
N_SEEDS = 5
TEST_FRAC = 0.2


"""loaders (ported inline; nothing added outside tests/)"""

def _stem(x):
    s = os.path.basename(str(x))
    while s.endswith(".wav"):
        s = s[:-4]
    return s.replace("audios-", "").replace("audios/", "")

def _name(a):
    return os.path.splitext(os.path.basename(str(a)))[0]

def load_neurovoz():
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
    sid = np.array([str(s).split("/")[-1].replace("audios-", "").replace(".wav", "")[-4:]
                    for s in common], dtype=object)
    ok = ~np.isnan(X).any(axis=1)
    return X[ok], Y[ok], sid[ok]


"""mnist-style generator (text.txt: y~p(y|u), x~p(x|u, pa(x)))"""

def generate(X_pool, Y_pool, quy, qyz, bright, seed_rand=1, seed_randn=2, frac=0.95):
    rng_r  = np.random.default_rng(seed_rand)
    rng_n  = np.random.default_rng(seed_randn)
    N_pool = X_pool.shape[0]
    perm = rng_r.permutation(N_pool)
    X_p, Y_p = X_pool[perm], Y_pool[perm]
    M = int(round(N_pool * frac))

    U = rng_r.binomial(1, 0.5, M).astype(np.float64)
    p1 = np.where(U == 1, quy, 1 - quy)
    Y = (rng_r.uniform(size=M) < p1).astype(np.float64)
    zsame = rng_r.uniform(size=M) < qyz
    Z = np.where(zsame, Y, 1 - Y).astype(int)

    X = np.empty((M, X_pool.shape[1]), dtype=np.float64)
    idx = {k: np.where(Y_p == k)[0] for k in (0, 1)}
    j = {k: 0 for k in (0, 1)}
    for n in range(M):
        k = Z[n]
        i = idx[k][j[k] % len(idx[k])]
        j[k] += 1
        X[n, :] = X_p[i] + U[n] * bright
    return X, Y, U


def tcb_weights_binary(U_conf, Y_conf, U_un, Y_un):
    """w_n = P_unconf(U_n) / (N * P_conf(U_n, Y_n)), mean-normalised."""
    N = len(U_conf)
    pu = {0: np.mean(U_un == 0), 1: np.mean(U_un == 1)}
    def puy(u, y):
        return np.mean((U_conf == u) & (Y_conf == y))
    w = np.array([pu[u] / (N * puy(u, y)) for u, y in zip(U_conf, Y_conf)])
    return w / w.mean()


def resample(X, Y, w, seed):
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(w), size=len(w), p=w / w.sum(), replace=True)
    return X[ix], Y[ix]


def two_sample_auc(A, B, seed=0):
    """Train LR to separate set A (label 0) from set B (label 1);
    AUC 0.5 => indistinguishable."""
    Xc = np.vstack([A, B])
    yc = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    rng = np.random.default_rng(seed)
    ix = rng.permutation(len(yc))
    Xc, yc = Xc[ix], yc[ix]
    m = int(0.7 * len(yc))
    clf = LogisticRegression(C=1e-3, max_iter=3000)
    clf.fit(Xc[:m], yc[:m])
    return roc_auc_score(yc[m:], clf.predict_proba(Xc[m:])[:, 1])


"""test suite"""

def make_sets(X, Y, sid, seed):
    subs = np.array(sorted(set(sid)))
    rng = np.random.default_rng(seed)
    te_subs = set(rng.permutation(subs)[:max(1, int(round(TEST_FRAC * len(subs))))])
    tr = np.array([s not in te_subs for s in sid])
    sc = StandardScaler().fit(X[tr])
    X_tr, X_te = sc.transform(X[tr]), sc.transform(X[~tr])
    Y_tr, Y_te = Y[tr], Y[~tr]

    conf_tr = generate(X_tr, Y_tr, QUY_CONF, QYZ, BRIGHT, seed_rand=seed*10+1, seed_randn=seed*10+2)
    un_tr   = generate(X_tr, Y_tr, 0.5,      QYZ, BRIGHT, seed_rand=seed*10+3, seed_randn=seed*10+4)
    un_te   = generate(X_te, Y_te, 0.5,      QYZ, BRIGHT, seed_rand=seed*10+5, seed_randn=seed*10+6)
    return conf_tr, un_tr, un_te


def test_weight_identity(conf_tr, un_tr, eps=0.02):
    """T1: reweighted confounded joint must equal the unconfounded joint.
    (eps covers finite-sample noise in the joint-cell estimates:
    max error ~ sqrt(p(1-p)/N) ~ 0.01 for N~2200.)"""
    Xc, Yc, Uc = conf_tr
    Xu, Yu, Uu = un_tr
    w = tcb_weights_binary(Uc, Yc, Uu, Yu)
    err = 0.0
    for u in (0, 1):
        for y in (0, 1):
            lhs = w[(Uc == u) & (Yc == y)].sum() / len(Uc)
            rhs = np.mean((Uu == u) & (Yu == y))
            err = max(err, abs(lhs - rhs))
    return err < eps, err


def test_recovery(conf_tr, un_tr, seed):
    """T2: TCB-resampled (X,Y) must match unconfounded (X,Y) per class."""
    Xc, Yc, Uc = conf_tr
    Xu, Yu, Uu = un_tr
    w = tcb_weights_binary(Uc, Yc, Uu, Yu)
    Xr, Yr = resample(Xc, Yc, w, seed)
    aucs = []
    for k in (0, 1):
        a = two_sample_auc(Xr[Yr == k], Xu[Yu == k], seed=seed)
        aucs.append(abs(a - 0.5))
    return max(aucs) < 0.08, aucs


def test_sufficiency(conf_tr, un_tr, un_te, seed):
    """T3: on the causal test set, U must add no label info beyond X."""
    Xc, Yc, Uc = conf_tr
    w = tcb_weights_binary(Uc, Yc, un_tr[1], un_tr[2])
    Xr, Yr = resample(Xc, Yc, w, seed)
    Xt, Yt, Ut = un_te

    def fit_auc(Xtr, ytr, Xte, yte):
        clf = LogisticRegression(C=1e-3, max_iter=3000).fit(Xtr, ytr)
        return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

    a_x   = fit_auc(Xr, Yr, Xt, Yt)
    a_xu  = fit_auc(np.column_stack([Xr, Uc]), Yr, np.column_stack([Xt, Ut]), Yt)
    return (a_xu - a_x) < 0.02, (a_x, a_xu)


def test_positivity(conf_tr, un_tr):
    """T4: effective sample size / overlap of the weights."""
    Xc, Yc, Uc = conf_tr
    w = tcb_weights_binary(Uc, Yc, un_tr[1], un_tr[2])
    ess = (w.sum() ** 2) / (w ** 2).sum()
    return ess / len(w) > 0.3, ess / len(w)


def test_mechanism(conf_tr, un_tr, un_te, seed):
    """T5: recovery must scale with qyz (causal path fidelity), and the
    deconfounded model must track the reference trained on causal data."""
    Xc, Yc, Uc = conf_tr
    w = tcb_weights_binary(Uc, Yc, un_tr[1], un_tr[2])
    Xr, Yr = resample(Xc, Yc, w, seed)
    Xt, Yt, _ = un_te

    def auc(Xtr, ytr):
        clf = LogisticRegression(C=1e-3, max_iter=3000).fit(Xtr, ytr)
        return roc_auc_score(Yt, clf.predict_proba(Xt)[:, 1])

    a_tcb = auc(Xr, Yr)
    a_ref = auc(un_tr[0], un_tr[1])
    return (a_tcb - a_ref) > -0.03, (a_tcb, a_ref)


def main():
    X, Y, sid = load_neurovoz()
    print(f"NeuroVoz: {X.shape[0]} samples, {len(set(sid))} speakers, {X.shape[1]} feats")
    print(f"confounded quy={QUY_CONF}, qyz={QYZ}, brightness={BRIGHT}\n")

    results = []
    for s in range(N_SEEDS):
        conf_tr, un_tr, un_te = make_sets(X, Y, sid, s)

        ok1, e1 = test_weight_identity(conf_tr, un_tr)
        ok2, a2 = test_recovery(conf_tr, un_tr, s)
        ok3, a3 = test_sufficiency(conf_tr, un_tr, un_te, s)
        ok4, e4 = test_positivity(conf_tr, un_tr)
        ok5, a5 = test_mechanism(conf_tr, un_tr, un_te, s)
        results.append((ok1, ok2, ok3, ok4, ok5, e1, a2, a3, e4, a5))
        print(f"seed {s}: "
              f"T1(err={e1:.4f}) T2(max|auc-.5|={max(a2):.3f}) "
              f"T3(Δauc={a3[1]-a3[0]:+.3f}) T4(ess={e4:.2f}) "
              f"T5(tcb={a5[0]:.3f}, ref={a5[1]:.3f})")

    n_ok = [sum(r[i] for r in results) for i in range(5)]
    print("\n--- verdict (X/{} seeds passed) ---".format(N_SEEDS))
    for name, n in zip(["T1 weight identity", "T2 recovery",
                        "T3 sufficiency (U adds no info)",
                        "T4 positivity", "T5 mechanism"], n_ok):
        print(f"  {name:<28} {n}/{N_SEEDS}")


if __name__ == "__main__":
    main()
