"""Figure 1(c): Z is a mediator between the cause y and the effect X.

Edges: y->X, y->Z, Z->X, y<->X (unobserved common effect), S->Z.
The target quantity is recovered by recalibrating P*(Z|y) class-wise
(disease duration: NeuroVoz "Time Disease (years)" vs. ParkCeleb upload
year - year of onset), so P*(X|do(y)) = sum_Z P(X|y,Z) P*(Z|y).
"""
import sys, os
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
for p in [str(BASE) + '/libraries/CausalBootstrapping', str(BASE) + '/libraries/GRAPL',
          str(BASE) + '/libraries/causal_assistant', str(BASE) + '/tCB', str(BASE) + '/tests']:
    if p not in sys.path:
        sys.path.insert(0, p)

import causal_assistant as ca
import sID_expr as sid_alg
import grapl.dsl as dsl
from scipy.stats import gaussian_kde, ks_2samp
from sklearn.ensemble import GradientBoostingClassifier

import transport_u_age as age

RANDOM_SEED = 42
N_BOOT = 50
N_AGE_BINS = 6
CB_FIT_METHOD = "kde"

META_COLS = {"Unnamed: 0", "tot_name"}
INTERVALS = {"-2": (-10, -6), "-1": (-5, -1), "1": (0, 5), "2": (6, 10)}
NV_FEAT = str(BASE) + '/data/neurovoz/extracted_features'

NAME = "Figure 1(c) -- Z=mediator between y and X; recalibrate P*(Z|y)"
SEL_DIAGRAM = "y;X;Z;S; y->X; y->Z; Z->X; y<->X; S->Z;"
CB_GRAPH = "y;X;Z; y->X; y->Z; Z->X;"
FORMULA = "P*(X|do(y)) = \\sum_Z P(X|do(y),Z) P*(Z|y)"


def _stem(x):
    s = str(x).rsplit("/", 1)[-1]
    while s.endswith(".wav"):
        s = s[:-4]
    return s.replace("audios-", "").replace("audios/", "")


def _sl(df, kv, common, meta_cols):
    return (df.loc[kv.isin(common)]
              .loc[:, ~df.columns.isin(meta_cols)]
              .set_index(kv[kv.isin(common)].values))


def load_nv():
    hc = pd.read_csv(str(BASE) + '/data/neurovoz/metadata/metadata_hc.csv')
    pd_ = pd.read_csv(str(BASE) + '/data/neurovoz/metadata/metadata_pd.csv')
    mt = pd.concat([hc, pd_], ignore_index=True)
    mt["_pid"] = mt["ID"].map(lambda i: f"{int(i):04d}")
    mt["Age"] = pd.to_numeric(mt["Age"], errors="coerce")
    a_by_pid = mt.groupby("_pid")["Age"].median()
    mt["UPDRS scale"] = pd.to_numeric(mt["UPDRS scale"], errors="coerce")
    u_by_pid = mt.groupby("_pid")["UPDRS scale"].median()
    mt["Time Disease (years)"] = pd.to_numeric(mt["Time Disease (years)"],
                                               errors="coerce")
    d_by_pid = mt.groupby("_pid")["Time Disease (years)"].median()

    grbas = {}
    for f in os.listdir(str(BASE) + '/data/grbas'):
        if not f.endswith(".csv"):
            continue
        for _, r in pd.read_csv(str(BASE) + '/data/grbas/' + f).iterrows():
            gid = os.path.splitext(os.path.basename(str(r["text_patient_id"])))[0].strip()
            try:
                grbas[gid] = float(r["G"])
            except (ValueError, TypeError):
                pass

    intens = pd.read_csv(str(NV_FEAT) + '/intensity_features_neurovoz.csv')
    pause = pd.read_csv(str(NV_FEAT) + '/pause_features_neurovoz.csv')
    syll = pd.read_csv(str(NV_FEAT) + '/syllable_features_neurovoz.csv')
    ki, kp, ks = (intens["tot_name"].apply(_stem),
                  pause["tot_name"].apply(_stem),
                  syll["tot_name"].apply(_stem))
    common = sorted(set(ki) & set(kp) & set(ks))
    X = pd.concat([_sl(intens, ki, common, META_COLS),
                   _sl(pause, kp, common, META_COLS),
                   _sl(syll, ks, common, META_COLS)], axis=1)
    X = X.values.astype(np.float64)
    Y = np.array([1 if c.startswith("PD_") else 0 for c in common], dtype=np.float64)
    pid = np.array([c[-4:] for c in common])
    ag = np.array([a_by_pid.get(p, np.nan) for p in pid], dtype=np.float64)
    up = np.array([np.nan if np.isnan(u_by_pid.get(p, np.nan)) else
                   u_by_pid.get(p, np.nan) for p in pid], dtype=np.float64)
    gr = np.array([grbas.get(c[3:], np.nan) for c in common], dtype=np.float64)
    dr = np.array([d_by_pid.get(p, np.nan) for p in pid], dtype=np.float64)
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(ag))
    return X[ok], Y[ok], ag[ok], pid[ok], up[ok], gr[ok], dr[ok]


def load_pc():
    base = str(BASE) + '/data/parkceleb_extracted'
    intens = pd.read_csv(str(base) + '/intensity_features_parkceleb.csv')
    pause = pd.read_csv(str(base) + '/pause_features_parkceleb.csv')
    syll = pd.read_csv(str(base) + '/syllable_features_parkceleb.csv')
    kid = intens["audio_file"].map(
        lambda f: (lambda m: m.group(1) if m else np.nan)(
            age.IDRE.search(str(f))))
    m = kid.notna()
    intens, pause, syll, kid = (intens[m].reset_index(drop=True),
                                pause[m].reset_index(drop=True),
                                syll[m].reset_index(drop=True),
                                kid[m].reset_index(drop=True))
    X = pd.concat([intens.loc[:, ~intens.columns.isin(age.FEATURE_META)],
                   pause.loc[:, ~pause.columns.isin(age.FEATURE_META)],
                   syll.loc[:, ~syll.columns.isin(age.FEATURE_META)]], axis=1)
    X = X.values.astype(np.float64)
    Y = intens["parkceleb_label"].values.astype(np.float64)
    sid = intens["speaker_id"].astype(str).values

    ages = pd.read_csv(str(BASE) + '/data/parkceleb_ages.csv')
    age_by_af = (ages.drop_duplicates("audio_file").set_index("audio_file")["age"])
    af = intens["audio_file"].astype(str).map(
        lambda f: f if f in age_by_af.index else np.nan)
    t_age = af.map(age_by_af).values.astype(np.float64)

    upload = pd.read_csv(str(BASE) + '/data/parkceleb_upload_years.csv')
    upload["video_id"] = upload["video_id"].astype(str)
    up_full = upload["upload_date"].astype(str).str[:4]
    up = up_full[up_full.str.fullmatch(r"\d{4}")].astype(int)
    up.index = upload.loc[up.index, "video_id"]

    import openpyxl
    wb = openpyxl.load_workbook(str(BASE) + '/data/ParkCeleb/PD_demo.xlsx',
                                read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    hdr = [str(c).strip().lower().replace(" ", "") for c in rows[0]]
    yod = {}
    for r in rows[2:]:
        if r[0] is None:
            continue
        yod[str(r[0]).strip()] = int(str(r[hdr.index("yod")]).strip())
    wb2 = openpyxl.load_workbook(str(BASE) + '/data/ParkCeleb/speakers_pairs.xlsx',
                                 read_only=True)
    rows2 = list(wb2.active.iter_rows(values_only=True))
    pairs = {}
    for r in rows2[1:]:
        if r[0] is None:
            continue
        pairs[str(r[0]).strip()] = str(r[1]).strip()

    iv = np.full(len(X), np.nan)
    for i, sp in enumerate(sid):
        vid = kid[i]
        if vid not in up.index:
            continue
        uy = up[vid]
        yr = yod.get(sp) if sp.startswith("pd_") else yod.get(pairs.get(sp))
        if yr is None:
            continue
        iv[i] = uy - yr

    out = {"full": (X, Y, sid, t_age, iv)}
    for name, (lo, hi) in INTERVALS.items():
        sel = (iv >= lo) & (iv <= hi)
        out[name] = (X[sel], Y[sel], sid[sel], t_age[sel], iv[sel])
    return out


def _ratio(U_s, U_t):
    U_s = np.asarray(U_s, dtype=float)
    U_t = np.asarray(U_t, dtype=float)
    U_t = U_t[np.isfinite(U_t)]
    if len(U_t) < 2 or np.std(U_t) < 1e-9 or np.std(U_s) < 1e-9:
        return np.ones(len(U_s))
    k_t, k_s = gaussian_kde(U_t), gaussian_kde(U_s)
    r = k_t(U_s) / np.maximum(k_s(U_s), 1e-12)
    lo, hi = np.percentile(r, 1), np.percentile(r, 99)
    return np.clip(r, lo, hi)


def _balanced_acc(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    return 0.5 * (np.mean(pred[y == 0] == 0) + np.mean(pred[y == 1] == 1))


def _gbc(X, y, w=None):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=3, random_state=RANDOM_SEED)
    if w is not None:
        w = np.asarray(w, dtype=float)
        w = w * (len(y) / np.maximum(w.sum(), 1e-12))
    clf.fit(np.asarray(X), y.astype(int), sample_weight=w)
    return clf


def _cb(X, y, z):
    kwargs = dict(causal_graph=CB_GRAPH, cause_var="y",
                  effect_var="X", fit_method=CB_FIT_METHOD,
                  X=X, y=y)
    zv = np.asarray(z, dtype=float)
    zv = np.where(np.isnan(zv), np.nanmedian(zv), zv)
    kwargs["Z"] = (zv, N_AGE_BINS)
    return np.asarray(ca.causal_weights(**kwargs))

def _cluster_sel(units, rng):
    u = np.unique(units)
    return np.isin(units, rng.choice(u, size=len(u), replace=True))


def _run_target(Xs, ys, pids, Xt, yt, r, rng, n_boot, z_src):
    cb_ba, raw_ba = [], []
    for _ in range(n_boot):
        s = _cluster_sel(pids, rng)
        zs_ = None if z_src is None else z_src[s]
        W = _cb(Xs[s], ys[s], zs_)
        w1 = np.maximum(W[:, 1] * r[s], 0.0)
        w1 = w1 + 1e-6 * np.maximum(w1.mean(), 1e-12)
        g_cb = _gbc(Xs[s], ys[s], w1)
        g_raw = _gbc(Xs[s], ys[s], None)
        cb_ba.append(_balanced_acc(yt, (g_cb.predict_proba(Xt)[:, 1] >= 0.5)))
        raw_ba.append(_balanced_acc(yt, (g_raw.predict_proba(Xt)[:, 1] >= 0.5)))
    return dict(cb=np.array(cb_ba), raw=np.array(raw_ba))


def main():
    print("sID transportability check ...", flush=True)
    try:
        g = dsl.GraplDSL().readgrapl(SEL_DIAGRAM)
        f = sid_alg.sid(Y={"X"}, X={"y"}, S={"S"}, D=g)
        print("   " + FORMULA, flush=True)
        print("   sID: " + (None if f is None else f.rhs.tocondstr()),
              flush=True)
    except Exception as e:
        print("   sID skipped:", type(e).__name__, e, flush=True)
    print()
    print("== " + NAME + " ==", flush=True)

    Xs, ys, ags, pids, _, _, drs_nv = load_nv()
    if ags.size > 0:
        age_str = 'age [' + str(int(np.nanmin(ags))) + '-' + str(int(np.nanmax(ags))) + ']'
    else:
        age_str = 'age [no data]'
    print('nv: ' + str(Xs.shape[0]) + ' recordings (' + str(int(ys.sum())) + ' pd / ' + str(len(ys) - int(ys.sum())) + ' hc), ' + age_str, flush=True)

    mu_s, sd_s = Xs.mean(0), Xs.std(0)
    sd_s = np.maximum(sd_s, np.median(sd_s) * 0.05)
    Xs = (Xs - mu_s) / sd_s

    pc = load_pc()
    zt = lambda X: (np.asarray(X, dtype=float) - mu_s) / sd_s

    targets = {}
    for name, (X, y, s, a, iv) in pc.items():
        m = ~np.isnan(X).any(axis=1)
        targets[name.upper()] = (zt(X[m]), y[m], s[m], a[m], iv[m])

    rng = np.random.RandomState(RANDOM_SEED)
    hdr = ('target n ba_c | ba_raw ks_p')
    print(hdr)

    ks_res = []
    Z_src = np.where(np.isnan(drs_nv), 0.0, drs_nv)
    for name, (Xt, yt, _, _, ivt) in targets.items():
        n_clips = len(yt)
        Xc = Xt

        r = np.ones(len(ys))
        for c in np.unique(ys):
            ms = ys == c
            mt = (yt == c)
            if ms.sum() > 1 and mt.sum() > 1 and np.std(ivt[mt]) > 1e-9:
                r[ms] = _ratio(Z_src[ms], ivt[mt])
        if np.any(np.isnan(r)):
            r = np.nan_to_num(r, nan=1.0)

        B = _run_target(Xs, ys, pids, Xc, yt, r, rng, N_BOOT, Z_src)
        cb_ba, raw_ba = B["cb"], B["raw"]

        p = 1.0
        if np.isfinite(cb_ba).all() and np.isfinite(raw_ba).all() \
                and np.std(cb_ba) + np.std(raw_ba) > 0:
            p = ks_2samp(cb_ba, raw_ba, alternative="less").pvalue

        print(str(name) + ' ' + str(n_clips) + ' ba_cb=' +
              str(np.median(cb_ba)) + ' ba_raw=' + str(np.median(raw_ba)) +
              ' ks_p=' + str(p), flush=True)
        ks_res.append(dict(target=name, p=p))

    print()
    print("KS (one-sided, p<=0.01): de-confounded BA > confounded BA on " +
          str(sum(1 for k in ks_res if k["p"] <= 0.01)) + "/" + str(len(ks_res)) +
          " targets")


main()