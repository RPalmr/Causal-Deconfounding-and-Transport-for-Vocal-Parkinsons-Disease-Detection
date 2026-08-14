import argparse, os
import numpy as np
import pandas as pd
import old.causal as c
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm
import openpyxl

BASE = c.BASE
OUT = str(BASE) + '/outputs'
QUALITY_CSV = str(BASE) + '/data/parkceleb_extracted/quality_features_parkceleb.csv'
N_FOLDS = 10
NV_EXCLUDED = {"0056", "0074", "0121", "0135"}

CONTROL_GRAPH = "y;X; y->X;"
AGE_PC = "y;X;Age; y->X; Age->X; Age->y;"
ACCENT_PC = "y;X;Accent; y->X; Accent->X; Accent->y;"
QUALITY_PC = "y;X;Quality; y->X; Quality->X; Quality->y;"
SEX_GRAPH = "y;X;Sex; y->X; Sex->X; Sex->y;"

NAT_CLUSTER = {
    "Canadian-America": 0, "British": 1, "English": 1, "Austrian-American": 2,
    "American-canadian": 0, "Irish": 2, "Italian": 2, "Spanish": 2, "Austrian": 2,
    "French": 2, "Romanian-born German": 2, "German": 2, "Scottish": 2,
    "Australian": 2, "Peruvian": 2, "Canadian": 0, "American": 0,
}

def _nt(v):
    return str(v).strip() if v is not None else ""

def _accent_map():
    import openpyxl
    acc = {}
    for path in (str(BASE) + '/data/ParkCeleb/PD_demo.xlsx',
                 str(BASE) + '/data/ParkCeleb/CN_demo.xlsx'):
        wb = openpyxl.load_workbook(path, read_only=True)
        rows = list(wb.active.iter_rows(values_only=True))
        hdr = [_nt(x).lower().replace(" ", "") for x in rows[0]]
        i_s = hdr.index("speakerid")
        i_n = hdr.index("nationality")
        i_e = hdr.index("ethnicity")
        for r in rows[1:]:
            if r[i_s] is None:
                continue
            sid = _nt(r[i_s]).lower()
            nat = _nt(r[i_n])
            eth = _nt(r[i_e])
            if nat not in NAT_CLUSTER:
                continue
            a = NAT_CLUSTER[nat]
            if eth and eth.lower() != "white":
                a += 2
            acc[sid] = int(a)
    return acc

def _load_quality():
    if not os.path.exists(QUALITY_CSV):
        return None
    return pd.read_csv(QUALITY_CSV)

def _weights(graph, X, y, feats, fit="kde"):
    kwargs = dict(causal_graph=graph, cause_var="y", effect_var="X",
                  fit_method=fit, X=X, y=y)
    for name, vals, bins, disc in feats:
        vals = np.asarray(vals, dtype=float)
        if disc and fit == "kde":
            r = np.random.default_rng(abs(hash(name)) % (2 ** 32))
            vals = vals + r.uniform(-0.05, 0.05, vals.shape[0] if vals.ndim == 1
                                    else vals.shape[0]).reshape(vals.shape)
        kwargs[name] = (vals, bins)
    W = np.asarray(c.ca.causal_weights(**kwargs))
    return W if np.isfinite(W).all() else np.ones(np.shape(W))

def _resample(X, y, W, pid, repeats):
    rng = np.random.default_rng(RANDOM_SEED)
    X, y, W = np.asarray(X), np.asarray(y), np.asarray(W, dtype=float)
    idx_all = []
    for d in (0, 1):
        m = y.astype(int) == d
        if not m.any():
            continue
        p = W[m, d] / W[m, d].sum()
        n_d = int(m.sum())
        for _ in range(repeats):
            idx_all.append(np.where(m)[0][rng.choice(n_d, n_d, p=p)])
    idx = np.concatenate(idx_all)
    return np.asarray(X)[idx], np.asarray(y)[idx], np.asarray(pid)[idx]

def _patient_masks(pid, y, n_folds, seed):
    pid = np.asarray(pid)
    y = np.asarray(y).astype(int)
    uniq = np.unique(pid)
    maj = np.array([int(np.bincount(y[pid == u]).argmax()) for u in uniq])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_i, te_i in skf.split(uniq, maj):
        yield np.isin(pid, uniq[tr_i]), np.isin(pid, uniq[te_i])

def _ks_p(a, b):
    a = np.asarray(a, dtype=float)[np.isfinite(a)]
    b = np.asarray(b, dtype=float)[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return 1.0
    return float(ks_2samp(a, b, alternative="greater").pvalue)

def _speaker_ba(sid, y, prob):
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob)
    yg, pg = [], []
    for s in np.unique(np.asarray(sid)):
        m = np.asarray(sid) == s
        if m.sum() == 0:
            continue
        yg.append(int(np.bincount(y[m]).argmax()))
        pg.append(prob[m].mean())
    return c._balanced_acc(np.array(yg), (np.array(pg) >= 0.5).astype(int))

def _load_keys(path, key_cols):
    if not os.path.exists(path):
        return set()
    try:
        d = pd.read_csv(path, dtype=str)
    except Exception:
        return set()
    if not pd.Index(key_cols).isin(d.columns).all():
        return set()
    return set(zip(*[d[k].values for k in key_cols]))

def _is_done(path, key_cols, vals):
    return tuple(str(v) for v in vals) in _load_keys(path, key_cols)

def _save_rows(path, key_cols, cols, rows):
    df = pd.DataFrame(rows, columns=cols)
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
    for k in key_cols:
        df[k] = df[k].astype(str)
    df = df.drop_duplicates(subset=key_cols, keep="last")
    df.to_csv(path, index=False)

def _rep_pc():
    print()
    print("section 1a: parkceleb replication (IFM3-equivalent GBC, per-file, patient 10-fold cv)")
    path = OUT + '/pipeline_replication_pc.csv'
    rows = []
    pc = c.load_pc()
    for name, (X_pc, y_pc, sid_pc, _) in pc.items():
        if name == "full" or len(np.unique(sid_pc)) < 8:
            continue
        if _is_done(path, ["interval"], [name]):
            print("  interval %-3s already saved, skipping" % name)
            continue
        ba_s = []
        for seed in range(RANDOM_SEED, RANDOM_SEED + 5):
            for tr, te in _patient_masks(sid_pc, y_pc, N_FOLDS, seed):
                sc = c._SS().fit(X_pc[tr])
                g = c._gbc(sc.transform(X_pc[tr]), y_pc[tr])
                ba_s.append(c._balanced_acc(
                    y_pc[te], (g.predict_proba(sc.transform(X_pc[te]))[:, 1] >= 0.5).astype(int)))
        ba = float(np.mean(ba_s))
        print("  interval %-3s n=%-4d speakers=%-3d  ba=%.4f" % (
            name, len(y_pc), len(np.unique(sid_pc)), ba))
        rows.append([name, len(y_pc), len(np.unique(sid_pc)), ba])
        _save_rows(path, ["interval"], ["interval", "n", "speakers", "ba"], rows)
    return pd.DataFrame(rows, columns=["interval", "n", "speakers", "ba"])

def _rep_nv():
    print()
    print("section 1b: neurovoz replication (RF / LR on provided vowel features, patient 10-fold cv)")
    df = pd.read_csv(str(BASE) + '/data/neurovoz/clean_data_featurized/full_data_processed_with_V.csv')
    df["user_id"] = df["user_id"].map(lambda s: str(s).zfill(4))
    df = df[~df["user_id"].isin(NV_EXCLUDED)]
    feats = [x for x in df.columns
             if x not in ("user_id", "word_id", "pk_label", "Sex", "Age")]
    ag = df.groupby("user_id")[feats + ["pk_label"]].median()
    X = ag[feats].values.astype(np.float64)
    y = ag["pk_label"].values.astype(int)
    pid = ag.index.values
    print("  patients=%d  pd=%d  hc=%d  features=%d" % (
        len(pid), int(y.sum()), len(y) - int(y.sum()), len(feats)))
    path = OUT + '/pipeline_replication_nv.csv'
    rows = []
    for name, model in (("random forest", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED)),
            ("logistic regression", LogisticRegression(
                l1_ratio=1, C=1.0, solver="saga", random_state=RANDOM_SEED,
                max_iter=10000))):
        if _is_done(path, ["model"], [name]):
            print("  %-20s  already saved, skipping" % name)
            continue
        ba_s = []
        for seed in range(RANDOM_SEED, RANDOM_SEED + 5):
            for tr, te in _patient_masks(pid, y, N_FOLDS, seed):
                sc = c._SS().fit(X[tr])
                model.fit(sc.transform(X[tr]), y[tr])
                p = model.predict_proba(sc.transform(X[te]))[:, 1]
                ba_s.append(c._balanced_acc(y[te], (p >= 0.5).astype(int)))
        ba = float(np.mean(ba_s))
        print("  %-20s  ba=%.4f" % (name, ba))
        rows.append([name, ba])
        _save_rows(path, ["model"], ["model", "ba"], rows)
    return pd.DataFrame(rows, columns=["model", "ba"])

def _nv_data():
    X, y, ag, sx, pid, up, gr, tasks = c.load_nv()
    m = ~np.isin(np.asarray(pid), list(NV_EXCLUDED))
    return X[m], y[m], ag[m], sx[m], pid[m], up[m], gr[m]

def _nv_weights(graph, X, y, ag, sx, up, gr):
    if graph == CONTROL_GRAPH:
        return np.ones((len(X), 2))
    feats = [("Age", ag, 6, False), ("Sex", sx, 2, True)]
    if "UPDRS" in graph:
        u = np.asarray(up, dtype=float)
        med = np.nanmedian(u[y.astype(int) == 1])
        med = med if np.isfinite(med) else 0.0
        u = np.where(np.isnan(u), np.where(y.astype(int) == 1, med, 0.0), u)
        feats.append(("UPDRS", u, 5, False))
    if "GRBAS" in graph:
        gv = np.asarray(gr, dtype=float)
        med = np.nanmedian(gv[y.astype(int) == 1])
        med = med if np.isfinite(med) else 0.0
        gv = np.where(np.isnan(gv), np.where(y.astype(int) == 1, med, 0.0), gv)
        feats.append(("GRBAS", gv, 4, False))
    return _weights(graph, X, y, feats)

def _same_corpus_nv(graph, variant, X, y, ag, sx, up, gr, pid, reps, repeats, label=None):
    W = _nv_weights(graph, X, y, ag, sx, up, gr)
    conf, dec = [], []
    for rep in tqdm(range(reps), desc=label, leave=False):
        for tr, te in _patient_masks(pid, y, N_FOLDS, RANDOM_SEED + rep):
            if variant == "weights":
                w = W[tr][np.arange(int(tr.sum())), y[tr].astype(int)]
                g_dc = c._gbc(X[tr], y[tr], w)
            else:
                Xd, yd, _ = _resample(X[tr], y[tr], W[tr], pid[tr], repeats)
                g_dc = c._gbc(Xd, yd)
            g_cf = c._gbc(X[tr], y[tr])
            conf.append(c._balanced_acc(
                y[te], (g_cf.predict_proba(X[te])[:, 1] >= 0.5).astype(int)))
            dec.append(c._balanced_acc(
                y[te], (g_dc.predict_proba(X[te])[:, 1] >= 0.5).astype(int)))
    return np.asarray(conf), np.asarray(dec)

def _same_corpus_pc(graph, fname, X, y, sid, vals, bins, reps, repeats, disc, label=None):
    W = _weights(graph, X, y, [(fname, vals, bins, disc)])
    conf, dec = [], []
    for rep in tqdm(range(reps), desc=label, leave=False):
        for tr, te in _patient_masks(sid, y, N_FOLDS, RANDOM_SEED + rep):
            w = W[tr][np.arange(int(tr.sum())), y[tr].astype(int)]
            g_dc = c._gbc(X[tr], y[tr], w)
            g_cf = c._gbc(X[tr], y[tr])
            conf.append(c._balanced_acc(
                y[te], (g_cf.predict_proba(X[te])[:, 1] >= 0.5).astype(int)))
            dec.append(c._balanced_acc(
                y[te], (g_dc.predict_proba(X[te])[:, 1] >= 0.5).astype(int)))
    return np.asarray(conf), np.asarray(dec)

def _conf_subsample(X, y, sex, p_m, p_f, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    sex = np.asarray(sex).astype(int)
    keep = np.zeros(len(y), dtype=bool)
    for g, p in ((1, p_m), (0, p_f)):
        hc = (sex == g) & (y == 0)
        pd = (sex == g) & (y == 1)
        keep[hc] = True
        n_hc = int(hc.sum())
        n_pd = min(int(round(n_hc * p / max(1 - p, 1e-9))), int(pd.sum()))
        if n_pd > 0:
            sel = np.where(pd)[0][rng.permutation(int(pd.sum()))[:n_pd]]
            keep[sel] = True
    i = np.where(keep)[0]
    return np.asarray(X)[i], y[i], sex[i], i

def _semisynth(X, y, sex, pid, repeats):
    X_c, y_c, s_c, i_c = _conf_subsample(X, y, sex, 0.8, 0.2, 101)
    X_r, y_r, s_r, i_r = _conf_subsample(X, y, sex, 0.2, 0.8, 202)
    pid_c = np.asarray(pid)[i_c]
    pid_r = np.asarray(pid)[i_r]
    W_c = _weights(SEX_GRAPH, X_c, y_c, [("Sex", s_c, 2, True)])
    X_dc, y_dc, pid_dc = _resample(X_c, y_c, W_c, pid_c, repeats)

    def loo_fit(case, Xtr, ytr, pidtr, Xte, yte):
        ba = []
        for u in tqdm(np.unique(pidtr), desc=case, leave=False):
            m = pidtr != u
            if len(np.unique(ytr[m])) < 2:
                continue
            g = c._gbc(Xtr[m], ytr[m])
            ba.append(c._balanced_acc(
                yte, (g.predict_proba(Xte)[:, 1] >= 0.5).astype(int)))
        return np.mean(ba)

    res = {}
    res["conf->conf"] = loo_fit("conf->conf", X_c, y_c, pid_c, X_c, y_c)
    res["conf->rev"] = loo_fit("conf->rev", X_c, y_c, pid_c, X_r, y_r)
    res["dec->conf"] = loo_fit("dec->conf", X_dc, y_dc, pid_dc, X_c, y_c)
    res["dec->rev"] = loo_fit("dec->rev", X_dc, y_dc, pid_dc, X_r, y_r)
    print("   confounded model    on confounded test    ba=%.4f" % res["conf->conf"])
    print("   confounded model    on reverse test       ba=%.4f" % res["conf->rev"])
    print("   deconfounded model  on confounded test    ba=%.4f" % res["dec->conf"])
    print("   deconfounded model  on reverse test       ba=%.4f" % res["dec->rev"])
    return res

def _transfer(pc, args):
    X_pb, y_pb, ag_pb, sx_pb, _, up_pb, gr_pb = c.load_nv(task="PETACA_BLANCA")
    feats = [("Age", ag_pb, 6, False), ("Sex", sx_pb, 2, True)]
    u = np.asarray(up_pb, dtype=float)
    med = np.nanmedian(u[y_pb.astype(int) == 1])
    med = med if np.isfinite(med) else 0.0
    u = np.where(np.isnan(u), np.where(y_pb.astype(int) == 1, med, 0.0), u)
    feats.append(("UPDRS", u, 5, False))
    gv = np.asarray(gr_pb, dtype=float)
    med = np.nanmedian(gv[y_pb.astype(int) == 1])
    med = med if np.isfinite(med) else 0.0
    gv = np.where(np.isnan(gv), np.where(y_pb.astype(int) == 1, med, 0.0), gv)
    feats.append(("GRBAS", gv, 4, False))
    pid_pb = np.arange(len(y_pb))

    from sklearn.decomposition import PCA
    mu_s, sd_s = X_pb.mean(0), np.maximum(X_pb.std(0), 1e-9)
    Xs = (X_pb - mu_s) / sd_s
    pca = PCA(n_components=2)
    L_nv = pca.fit_transform(Xs[:, c.INTENSITY_IDX])
    corrs = np.array([abs(np.corrcoef(L_nv[:, i], y_pb)[0, 1]) for i in range(2)])
    k_u = np.argsort(corrs)[:c.U_DIMS]
    U_nv = L_nv[:, k_u]

    print()
    print("section 5: cross-corpora transport nv -> parkceleb intervals (density-ratio reweighting)")
    print("   nv model transported to each parkceleb interval: source samples reweighted")
    print("   by p(u | parkceleb) / p(u | nv), u = intensity pcs (u is not a cause of y);")
    print("   empirical transport under covariate shift, not formal selection-diagram")
    print("   transportability (Bareinboim and Pearl)")
    path = OUT + '/pipeline_transfer.csv'
    rows = []
    for name, (X_pc, y_pc, sid_pc, _) in pc.items():
        if name == "full":
            continue
        if _is_done(path, ["interval", "reps", "cb_repeats"],
                    [name, args.xfer_reps, args.cb_repeats]):
            print("  %-3s  already saved, skipping" % name)
            continue
        m0 = ~np.isnan(X_pc).any(axis=1)
        X_t = (X_pc[m0] - mu_s) / sd_s
        y_t, sid_t = y_pc[m0], sid_pc[m0]
        U_pc = pca.transform(X_t[:, c.INTENSITY_IDX])[:, k_u]
        w_t = c._tcb_weights(U_pc, U_nv, y_pb)
        W_t = np.column_stack([w_t, w_t])
        med = {"conf": [], "dec": []}
        for _ in tqdm(range(args.xfer_reps), desc="interval %s" % name, leave=False):
            Xd, yd, _ = _resample(X_pb, y_pb, W_t, pid_pb, args.cb_repeats)
            g_dc = c._gbc(Xd, yd)
            g_cf = c._gbc(X_pb, y_pb)
            med["dec"].append(_speaker_ba(sid_t, y_t, g_dc.predict_proba(X_t)[:, 1]))
            med["conf"].append(_speaker_ba(sid_t, y_t, g_cf.predict_proba(X_t)[:, 1]))
        mc, md = float(np.median(med["conf"])), float(np.median(med["dec"]))
        p = _ks_p(med["dec"], med["conf"])
        print("  %-3s  conf=%.4f  deconf=%.4f  dacc=%+.4f  p_ks=%.4f" % (
            name, mc, md, md - mc, p))
        rows.append([name, mc, md, md - mc, p, args.xfer_reps, args.cb_repeats])
        _save_rows(path, ["interval"],
                   ["interval", "conf", "deconf", "dacc", "p_ks",
                    "reps", "cb_repeats"], rows)
    return pd.DataFrame(rows, columns=["interval", "conf", "deconf", "dacc",
                                       "p_ks", "reps", "cb_repeats"])

def main():
    global RANDOM_SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--pc-reps", type=int, default=50)
    ap.add_argument("--nv-reps", type=int, default=200)
    ap.add_argument("--xfer-reps", type=int, default=200)
    ap.add_argument("--cb-repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=c.RANDOM_SEED)
    args = ap.parse_args()
    RANDOM_SEED = args.seed
    if args.reps is not None:
        args.pc_reps = args.nv_reps = args.xfer_reps = args.reps

    os.makedirs(OUT, exist_ok=True)

    _rep_pc()
    _rep_nv()

    print()
    print("section 2: causal variable identification")
    acc_map = _accent_map()
    pc = c.load_pc()
    Xf, yf, sidf, agef = pc["full"]
    accent_all = np.array([acc_map.get(s, np.nan) for s in sidf], dtype=float)
    ok_a = ~np.isnan(accent_all)
    if ok_a.any():
        print("   parkceleb: accent available for %d/%d clips (%d speakers), clusters=%s" % (
            int(ok_a.sum()), len(yf), len(np.unique(sidf[ok_a])),
            sorted(np.unique(accent_all[ok_a]).astype(int).tolist())))
    q = _load_quality()
    if q is None:
        print("   parkceleb: audio quality scores unavailable (expected: "
              "data/parkceleb_extracted/quality_features_parkceleb.csv, keyed by "
              "speaker_id with quality columns) - skipped")
    else:
        print("   parkceleb: audio quality scores loaded (%d clips)" % len(q))

    X, y, ag, sx, pid, up, gr = _nv_data()
    print("   neurovoz: Age pd=%.1f hc=%.1f | Sex pd=%.1f hc=%.1f | UPDRS pd=%.1f | GRBAS mean pd=%.2f hc=%.2f" % (
        ag[y.astype(int) == 1].mean(), ag[y.astype(int) == 0].mean(),
        sx[y.astype(int) == 1].mean(), sx[y.astype(int) == 0].mean(),
        np.nanmedian(up[y.astype(int) == 1]),
        np.nanmean(gr[y.astype(int) == 1]), np.nanmean(gr[y.astype(int) == 0])))

    print()
    print("section 3: cb de-confounding, two variants (bootstrap resample vs direct weights)")
    path3 = OUT + '/pipeline_variants_nv.csv'
    rows3 = []
    for graph in ("control", "ug-as[a]"):
        gx = CONTROL_GRAPH if graph == "control" else c.GRAPH_MODELS[graph]
        for variant in ("bootstrap", "weights"):
            if _is_done(path3, ["graph", "variant", "reps", "cb_repeats"],
                        [graph, variant, args.nv_reps, args.cb_repeats]):
                print("   nv %-8s %-9s  already saved, skipping" % (graph, variant))
                continue
            conf, dec = _same_corpus_nv(gx, variant, X, y, ag, sx, up, gr, pid,
                                        args.nv_reps, args.cb_repeats,
                                        label="nv %s %s" % (graph, variant))
            mconf, mdec = float(np.median(conf)), float(np.median(dec))
            p = _ks_p(dec, conf)
            print("   nv %-8s %-9s  conf=%.4f  deconf=%.4f  dacc=%+.4f  p_ks=%.4f" % (
                graph, variant, mconf, mdec, mdec - mconf, p))
            rows3.append([graph, variant, mconf, mdec, mdec - mconf, p,
                          args.nv_reps, args.cb_repeats])
            _save_rows(path3, ["graph", "variant"],
                       ["graph", "variant", "conf", "deconf", "dacc", "p_ks",
                        "reps", "cb_repeats"], rows3)

    print()
    print("section 4a: same-corpus evaluation, parkceleb back-door variables")
    path4 = OUT + '/pipeline_samecorpus_pc.csv'
    rows4 = []
    if len(np.unique(sidf[~np.isnan(agef)])) >= 8:
        m = ~np.isnan(agef)
        if not _is_done(path4, ["variable", "reps", "cb_repeats"],
                        ["age", args.pc_reps, args.cb_repeats]):
            conf, dec = _same_corpus_pc(AGE_PC, "Age", Xf[m], yf[m], sidf[m],
                                        agef[m], 6, args.pc_reps, args.cb_repeats,
                                        False, label="pc age")
            mconf, mdec = float(np.median(conf)), float(np.median(dec))
            p = _ks_p(dec, conf)
            print("   pc age      n=%-4d conf=%.4f  deconf=%.4f  dacc=%+.4f  p_ks=%.4f" % (
                int(m.sum()), mconf, mdec, mdec - mconf, p))
            rows4.append(["age", int(m.sum()), mconf, mdec, mdec - mconf, p,
                          args.pc_reps, args.cb_repeats])
            _save_rows(path4, ["variable"],
                       ["variable", "n", "conf", "deconf", "dacc", "p_ks",
                        "reps", "cb_repeats"], rows4)
        else:
            print("   pc age      already saved, skipping")
    if ok_a.any() and len(np.unique(sidf[ok_a])) >= 8:
        n_cl = len(np.unique(accent_all[ok_a]))
        if not _is_done(path4, ["variable", "reps", "cb_repeats"],
                        ["accent", args.pc_reps, args.cb_repeats]):
            conf, dec = _same_corpus_pc(ACCENT_PC, "Accent", Xf[ok_a], yf[ok_a],
                                        sidf[ok_a], accent_all[ok_a], n_cl,
                                        args.pc_reps, args.cb_repeats, True,
                                        label="pc accent")
            mconf, mdec = float(np.median(conf)), float(np.median(dec))
            p = _ks_p(dec, conf)
            print("   pc accent  n=%-4d conf=%.4f  deconf=%.4f  dacc=%+.4f  p_ks=%.4f" % (
                int(ok_a.sum()), mconf, mdec, mdec - mconf, p))
            rows4.append(["accent", int(ok_a.sum()), mconf, mdec, mdec - mconf, p,
                          args.pc_reps, args.cb_repeats])
            _save_rows(path4, ["variable"],
                       ["variable", "n", "conf", "deconf", "dacc", "p_ks",
                        "reps", "cb_repeats"], rows4)
        else:
            print("   pc accent  already saved, skipping")
    if q is not None:
        qcols = [x for x in q.columns
                 if x not in ("video_id", "audio_file", "speaker_id")]
        qd = {str(k): np.asarray(v, dtype=float)
              for k, v in zip(q["video_id"].astype(str), q[qcols].values)}
        intens = pd.read_csv(str(BASE) +
                             '/data/parkceleb_extracted/intensity_features_parkceleb.csv')
        kid = intens["audio_file"].map(
            lambda f: (lambda mm: mm.group(1) if mm else np.nan)(
                c.age.IDRE.search(str(f))))
        mk = kid.notna().values
        kidr = kid[mk].values
        sidr = intens.loc[mk, "speaker_id"].astype(str).values
        upload = pd.read_csv(str(BASE) + '/data/parkceleb_upload_years.csv')
        up_full = upload["upload_date"].astype(str).str[:4]
        uy = up_full[up_full.str.fullmatch(r"\d{4}")].astype(int)
        uy.index = upload.loc[uy.index, "video_id"]
        uy = uy.to_dict()
        wb = openpyxl.load_workbook(str(BASE) + '/data/ParkCeleb/PD_demo.xlsx',
                                    read_only=True)
        rows = list(wb.active.iter_rows(values_only=True))
        hdr = [str(c).strip().lower().replace(" ", "") for c in rows[0]]
        yod = {}
        for r in rows[2:]:
            if r[0] is None:
                continue
            yod[str(r[0]).strip()] = int(str(r[hdr.index("yod")]).strip())
        wb2 = openpyxl.load_workbook(str(BASE) +
                                     '/data/ParkCeleb/speakers_pairs.xlsx',
                                     read_only=True)
        rows2 = list(wb2.active.iter_rows(values_only=True))
        pairs = {}
        for r in rows2[1:]:
            if r[0] is None:
                continue
            pairs[str(r[0]).strip()] = str(r[1]).strip()
        iv = np.full(len(kidr), np.nan)
        for i, (sp, vid) in enumerate(zip(sidr, kidr)):
            if vid not in uy:
                continue
            yr = yod.get(sp) if sp.startswith("pd_") else yod.get(pairs.get(sp))
            if yr is None:
                continue
            iv[i] = uy[vid] - yr
        cent = np.zeros(len(kidr), dtype=bool)
        for lo, hi in c.INTERVALS.values():
            cent |= (iv >= lo) & (iv <= hi)
        qv = np.array([qd.get(str(v), np.full(len(qcols), np.nan))
                       for v in kidr])
        mq = cent & ~np.isnan(qv).any(axis=1)
        if not _is_done(path4, ["variable", "reps", "cb_repeats"],
                        ["quality", args.pc_reps, args.cb_repeats]):
            conf, dec = _same_corpus_pc(QUALITY_PC, "Quality", Xf[mq], yf[mq],
                                        sidf[mq], qv[mq], [5] * len(qcols),
                                        args.pc_reps, args.cb_repeats, False,
                                        label="pc quality")
            mconf, mdec = float(np.median(conf)), float(np.median(dec))
            p = _ks_p(dec, conf)
            print("   pc quality n=%-4d conf=%.4f  deconf=%.4f  dacc=%+.4f  p_ks=%.4f" % (
                int(mq.sum()), mconf, mdec, mdec - mconf, p))
            rows4.append(["quality", int(mq.sum()), mconf, mdec, mdec - mconf, p,
                          args.pc_reps, args.cb_repeats])
            _save_rows(path4, ["variable"],
                       ["variable", "n", "conf", "deconf", "dacc", "p_ks",
                        "reps", "cb_repeats"], rows4)
        else:
            print("   pc quality already saved, skipping")
        sco = []
        for j in range(len(qcols)):
            ba_s = []
            for tr, te in _patient_masks(sidf[mq], yf[mq], N_FOLDS, RANDOM_SEED):
                svc = SVC(C=1.0).fit(qv[mq][tr, j:j + 1], yf[mq].astype(int)[tr])
                ba_s.append(c._balanced_acc(
                    yf[mq].astype(int)[te], svc.predict(qv[mq][te, j:j + 1])))
            sco.append(float(np.mean(ba_s)))
        jb = int(np.argmax(sco))
        if not _is_done(path4, ["variable", "reps", "cb_repeats"],
                        ["quality-" + qcols[jb], args.pc_reps, args.cb_repeats]):
            conf, dec = _same_corpus_pc(QUALITY_PC, "Quality",
                                        Xf[mq], yf[mq], sidf[mq], qv[mq][:, jb:jb + 1],
                                        [10], args.pc_reps, args.cb_repeats, False,
                                        label="pc quality-" + qcols[jb])
            mconf, mdec = float(np.median(conf)), float(np.median(dec))
            p = _ks_p(dec, conf)
            print("   pc quality-top1 (%-6s, svc ba=%.3f)  n=%-4d conf=%.4f  deconf=%.4f  dacc=%+.4f  p_ks=%.4f" % (
                qcols[jb], sco[jb], int(mq.sum()), mconf, mdec, mdec - mconf, p))
            rows4.append(["quality-" + qcols[jb], int(mq.sum()), mconf, mdec,
                          mdec - mconf, p, args.pc_reps, args.cb_repeats])
            _save_rows(path4, ["variable"],
                       ["variable", "n", "conf", "deconf", "dacc", "p_ks",
                        "reps", "cb_repeats"], rows4)
        else:
            print("   pc quality-top1 already saved, skipping")
    elif q is None and not _is_done(path4, ["variable", "reps", "cb_repeats"],
                                    ["quality", args.pc_reps, args.cb_repeats]):
        rows4.append(["quality", 0, np.nan, np.nan, np.nan, np.nan,
                      args.pc_reps, args.cb_repeats])

    print()
    print("section 4b: same-corpus evaluation, neurovoz 'g-as[a]' (headline graph)")
    path4b = OUT + '/pipeline_samecorpus_nv.csv'
    if not _is_done(path4b, ["graph", "reps", "cb_repeats"],
                    ["g-as[a]", args.nv_reps, args.cb_repeats]):
        conf, dec = _same_corpus_nv(c.GRAPH_MODELS["g-as[a]"], "bootstrap",
                                    X, y, ag, sx, up, gr, pid, args.nv_reps,
                                    args.cb_repeats, label="nv g-as[a]")
        mconf, mdec = float(np.median(conf)), float(np.median(dec))
        p = _ks_p(dec, conf)
        print("   nv g-as[a]  conf=%.4f  deconf=%.4f  dacc=%+.4f  p_ks=%.4f" % (
            mconf, mdec, mdec - mconf, p))
        _save_rows(path4b, ["graph"],
                   ["graph", "conf", "deconf", "dacc", "p_ks",
                    "reps", "cb_repeats"],
                   [["g-as[a]", mconf, mdec, mdec - mconf, p,
                     args.nv_reps, args.cb_repeats]])
    else:
        print("   nv g-as[a]  already saved, skipping")

    print()
    print("section 4c: semi-synthetic sanity check (gender-confounded neurovoz, LOO)")
    path4c = OUT + '/pipeline_semisynth.csv'
    if not all(_is_done(path4c, ["case", "cb_repeats"], [case, args.cb_repeats])
               for case in ("conf->conf", "conf->rev", "dec->conf", "dec->rev")):
        res = _semisynth(X, y, sx, pid, args.cb_repeats)
        _save_rows(path4c, ["case"],
                   ["case", "ba", "cb_repeats"],
                   [[case, res[case], args.cb_repeats] for case in res])
    else:
        print("   all cases already saved, skipping")

    _transfer(pc, args)
    print()
    print("done: outputs in " + OUT)

if __name__ == "__main__":
    main()