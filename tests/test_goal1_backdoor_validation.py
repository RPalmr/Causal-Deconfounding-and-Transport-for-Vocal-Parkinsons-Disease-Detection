"""
Goal 1: within-dataset back-door validation of the causal-bootstrapping
        weighting mechanism (live from Cartlidge's library)
================================================================================
Purpose
-------
Validate the weighting mechanism against a ground truth we build ourselves,
BEFORE any real cross-corpus transport (Goal 2). Every surprising result so
far has been a formula mismatch or leakage bug; this is the check that would
have caught them, because here we know the right answer.

Design (MNIST mechanism / text.txt, fully synthetic U, NEVER touches ParkCeleb)
-------------------------------------------------------------------------------
  - real NeuroVoz features X (causal signal: PD -> voice)
  - synthetic binary U injected per text.txt:
        Y ~ P(Y|U) via quy,  X = X_real[Z] + U*brightness,  Z ~ P(Z|Y) via qyz
  - three self-constructed datasets, all NV-based:
        confounded      quy = 0.85   (U drives Y: polluted)
        non-confounded  quy = 0.50   (U independent of Y: ground truth)
        reverse-confounded quy = 0.15 (U anti-correlated with Y)
  - weights come from the LIBRARY, live: general_cb_analysis on the graph
        y -> X,  y <- U -> X
    which identifies the within-dataset back-door adjustment
        p_y(X) = sum_U p(X|y,U) p(U)   ->   w_n = P(U_n)K[y_i,y_n] / [N P(U_n,y_i)]
    computed with the library's own build_weight_function + weight_compute
    (exact-count PMFs, equality kernel). NOT hand-derived, NOT the transport
    formula: this is one dataset, one known measured confounder.

Checks
------
  G1  formula   : library identifies the back-door structure (P(U), P(U,y), kernel)
  G2  identity  : reweighted confounded P(U,Y) equals the non-confounded P(U,Y)
  G3  ground truth: weighted-resampled GBC ~= oracle (trained on non-confounded)
                   on ALL THREE eval sets (confounded / non / reverse)
  G4  pollution : plain GBC tracks the confound (inflated on confounded eval,
                  sign-reversed on reverse eval) - the failure G3 must not have

Model: ordinary GradientBoostingClassifier on Cartlidge-style weighted
resampling (replacement, weights proportional, R repetitions averaged).
"""
import sys, os, warnings
import inspect
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ -> root
DATA = f"{BASE}/data"
for p in [f"{BASE}/libraries/GRAPL", f"{BASE}/libraries/CausalBootstrapping"]:
    if p not in sys.path:
        sys.path.insert(0, p)
from causalbootstrapping.workflows import general_cb_analysis
from causalbootstrapping.backend import weight_compute

N_FEATS = 66
QUY_CONF = 0.85
QUY_NON = 0.50
QUY_REV = 0.15
QYZ = 0.95
BRIGHT = 2.0
N_SEEDS = 5
TEST_FRAC = 0.2
R_REPS = 3


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
    pause = pd.read_csv(f"{base}/pause_features_neurovoz.csv")
    syll = pd.read_csv(f"{base}/syllable_features_neurovoz.csv")
    meta_cols = {"Unnamed: 0", "tot_name"}
    ki = intens["tot_name"].apply(_stem)
    kp = pause["tot_name"].apply(_stem)
    ks = syll["tot_name"].apply(_stem)
    common = sorted(set(ki) & set(kp) & set(ks))
    def sl(df, kv):
        return df.loc[kv.isin(common)].loc[:, ~df.columns.isin(meta_cols)].set_index(kv[kv.isin(common)].values)
    X = pd.concat([sl(intens, ki), sl(pause, kp), sl(syll, ks)], axis=1).values.astype(np.float64)
    Y = np.array([1 if s.startswith("PD_") else 0 for s in common], dtype=np.float64)
    sid = np.array([str(s).split("/")[-1].replace("audios-", "").replace(".wav", "")[-4:]
                    for s in common], dtype=object)
    ok = ~np.isnan(X).any(axis=1)
    return X[ok], Y[ok], sid[ok]


"""mnist-style generator (text.txt: y~p(y|u), x~p(x|u, pa(x)))"""

def generate(X_pool, Y_pool, quy, qyz, bright, seed_rand=1, seed_randn=2, frac=0.95):
    """U acts on a dedicated synthetic feature column (MNIST-background
    equivalent, disjoint from the real signal channel), never on the real
    acoustic features themselves."""
    rng_r = np.random.default_rng(seed_rand)
    rng_n = np.random.default_rng(seed_randn)
    N_pool = X_pool.shape[0]
    perm = rng_r.permutation(N_pool)
    X_p, Y_p = X_pool[perm], Y_pool[perm]
    M = int(round(N_pool * frac))

    U = rng_r.binomial(1, 0.5, M).astype(np.float64)
    p1 = np.where(U == 1, quy, 1 - quy)
    Y = (rng_r.uniform(size=M) < p1).astype(np.float64)
    zsame = rng_r.uniform(size=M) < qyz
    Z = np.where(zsame, Y, 1 - Y).astype(int)

    X = np.empty((M, X_pool.shape[1] + 1), dtype=np.float64)
    idx = {k: np.where(Y_p == k)[0] for k in (0, 1)}
    j = {k: 0 for k in (0, 1)}
    for n in range(M):
        k = Z[n]
        i = idx[k][j[k] % len(idx[k])]
        j[k] += 1
        X[n, :-1] = X_p[i]
    X[:, -1] = U * bright + rng_n.normal(0.0, 0.5, M)
    return X, Y, U


"""back-door weights, live from cartlidge's library"""

def exact_pmf(names, vals):
    """Exact-count PMF with positional-or-keyword params named after the
    graph variables (the library calls dist functions by parameter name)."""
    vals2 = vals.reshape(-1, len(names))
    def _core(*args):
        arrs = [np.atleast_1d(a).reshape(-1) for a in args]
        X = np.column_stack(arrs)
        return np.array([np.mean((vals2 == row[None, :]).all(axis=1)) for row in X])
    ns = {"_core": _core}
    exec(f"_pdf = lambda {','.join(names)}: _core({','.join(names)})", ns)
    return ns["_pdf"]


def library_backdoor_weights(U, Y, info_print=False):
    """Within-dataset back-door weights for the graph y -> X, y <- U -> X,
    computed entirely by the library: general_cb_analysis identifies the
    formula, build_weight_function + weight_compute evaluate it.

    Returns (w, formula_dict): per-sample weight for the sample's OWN y
    (equality kernel), plus the library's identified formula structure."""
    N = len(U)
    graph = "y;X;U;\ny->X;\nU->y;\nU->X;\n"
    wfl, wfs = general_cb_analysis(causal_graph=graph, effect_var_name="X",
                                   cause_var_name="y", info_print=info_print)
    w_nom = sorted(s for term in wfs.w_nom for s in term)
    w_denom = sorted(s for term in wfs.w_denom for s in term)
    p_uy = exact_pmf(["U", "y"], np.column_stack([U, Y]))
    dist_map = {"U": exact_pmf(["U"], U), "U,y": p_uy, "U,intv_y": p_uy}
    kernel = lambda intv_y, y: np.where(np.equal(intv_y, y).ravel(), 1.0, 0.0)
    wbuilt = wfl(dist_map=dist_map, N=N, kernel=kernel,
                 cause_intv_name_map={"y": "intv_y"})
    wfunc, _ = wbuilt if isinstance(wbuilt, tuple) else (wbuilt, None)
    cols = np.stack([weight_compute(wfunc, {"U": U.reshape(-1, 1), "y": Y.reshape(-1, 1)},
                                    {"intv_y": [yv]}) for yv in (0, 1)]).T
    w_own = np.choose(Y.astype(int), cols.T)
    return w_own, {"w_nom": w_nom, "w_denom": w_denom,
                   "kernel_flag": wfs.kernel_flag}


"""cartlidge-style training: weighted resampling + ordinary gbc"""

def resample(X, Y, w, seed):
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(w), size=len(w), p=w / w.sum(), replace=True)
    return X[ix], Y[ix]


def fit_auc(Xtr, ytr, Xte, yte, seed):
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.1, random_state=seed)
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def weighted_auc(Xtr, ytr, w, Xte, yte, seed, reps=R_REPS):
    """Cartlidge step 4: resample repeatedly, average predicted probabilities."""
    probs = np.zeros(len(Xte))
    for r in range(reps):
        Xr, yr = resample(Xtr, ytr, w, seed * 100 + r)
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         learning_rate=0.1, random_state=seed * 100 + r)
        clf.fit(Xr, yr)
        probs += clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, probs / reps)


"""checks"""

def make_sets(X, Y, sid, seed):
    subs = np.array(sorted(set(sid)))
    rng = np.random.default_rng(seed)
    te_subs = set(rng.permutation(subs)[:max(1, int(round(TEST_FRAC * len(subs))))])
    tr = np.array([s not in te_subs for s in sid])
    sc = StandardScaler().fit(X[tr])
    X_tr, X_te = sc.transform(X[tr]), sc.transform(X[~tr])
    Y_tr, Y_te = Y[tr], Y[~tr]
    conf_tr = generate(X_tr, Y_tr, QUY_CONF, QYZ, BRIGHT, seed_rand=seed * 10 + 1, seed_randn=seed * 10 + 2)
    un_tr   = generate(X_tr, Y_tr, QUY_NON,  QYZ, BRIGHT, seed_rand=seed * 10 + 3, seed_randn=seed * 10 + 4)
    eval_sets = {
        "confounded":          generate(X_te, Y_te, QUY_CONF, QYZ, BRIGHT, seed_rand=seed * 10 + 5, seed_randn=seed * 10 + 6, frac=1.0),
        "non-confounded":      generate(X_te, Y_te, QUY_NON,  QYZ, BRIGHT, seed_rand=seed * 10 + 7, seed_randn=seed * 10 + 8, frac=1.0),
        "reverse-confounded":  generate(X_te, Y_te, QUY_REV,  QYZ, BRIGHT, seed_rand=seed * 10 + 9, seed_randn=seed * 10 + 10, frac=1.0),
    }
    return conf_tr, un_tr, eval_sets


def check_formula(f):
    ok = (f["w_nom"] == ["U"] and f["w_denom"] == ["U", "y"] and f["kernel_flag"])
    return ok, f


def check_identity(conf_tr, un_tr, w):
    """G2: mean-normalised weighted mass per (U,Y) cell == non-confounded freq."""
    Xc, Yc, Uc = conf_tr
    Xu, Yu, Uu = un_tr
    w = w / w.mean()
    err = 0.0
    for u in (0, 1):
        for y in (0, 1):
            lhs = w[(Uc == u) & (Yc == y)].sum() / len(Uc)
            rhs = np.mean((Uu == u) & (Yu == y))
            err = max(err, abs(lhs - rhs))
    return err < 0.03, err


def main():
    X, Y, sid = load_neurovoz()
    print(f"NeuroVoz: {X.shape[0]} samples, {len(set(sid))} speakers, {X.shape[1]} feats")
    print(f"confounded quy={QUY_CONF}, non-confounded quy={QUY_NON}, "
          f"reverse quy={QUY_REV}, qyz={QYZ}, brightness={BRIGHT}\n")

    g1_ok = g2_ok = g3_ok = g4_ok = 0
    for s in range(N_SEEDS):
        conf_tr, un_tr, ev = make_sets(X, Y, sid, s)
        Xc, Yc, Uc = conf_tr

        w, f = library_backdoor_weights(Uc, Yc, info_print=(s == 0))
        ok1, f1 = check_formula(f)
        ok2, e2 = check_identity(conf_tr, un_tr, w)

        a_plain, a_w, a_oracle = {}, {}, {}
        for name, (Xt, Yt, Ut) in ev.items():
            a_plain[name] = fit_auc(Xc, Yc, Xt, Yt, s)
            a_w[name] = weighted_auc(Xc, Yc, w, Xt, Yt, s)
            a_oracle[name] = fit_auc(un_tr[0], un_tr[1], Xt, Yt, s)

        ok3 = all(abs(a_w[n] - a_oracle[n]) < 0.03 for n in ev)
        ok4 = ((a_plain["confounded"] - a_oracle["confounded"]) > 0.02
               and (a_oracle["reverse-confounded"] - a_plain["reverse-confounded"]) > 0.02)
        g1_ok += ok1; g2_ok += ok2; g3_ok += ok3; g4_ok += ok4

        print(f"seed {s}: G1={int(ok1)} G2(err={e2:.4f})")
        for n in ev:
            print(f"    {n:<19} plain={a_plain[n]:.3f} weighted={a_w[n]:.3f} oracle={a_oracle[n]:.3f}")
        print(f"    G3(weighted~oracle on all 3)={int(ok3)}  G4(pollution present)={int(ok4)}\n")

    print("--- verdict (X/{} seeds passed) ---".format(N_SEEDS))
    for name, n in zip(["G1 library formula is back-door",
                        "G2 reweighted joint = unconfounded",
                        "G3 weighted ~= oracle on all 3 eval sets",
                        "G4 plain model shows the confound"], [g1_ok, g2_ok, g3_ok, g4_ok]):
        print(f"  {name:<46} {n}/{N_SEEDS}")


if __name__ == "__main__":
    main()
