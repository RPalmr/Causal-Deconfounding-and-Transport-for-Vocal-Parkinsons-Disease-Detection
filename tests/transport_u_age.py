"""stage-2 raw-feature transport experiment for the dissertation's raw-feature chapter.

trains a neurovoz gradient-boosted classifier with u=age appended and
evaluates cross-corpus transfer to parkceleb under transported, sID-
weighted, feature-aligned, placebo, no-u, and age-only variants."""

import sys, os
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import gaussian_kde

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{BASE}/data"
for p in [f"{BASE}/libraries/GRAPL", f"{BASE}/libraries/CausalBootstrapping",
          f"{BASE}/tCB", BASE]:
    if p not in sys.path:
        sys.path.insert(0, p)

import transport_u_duration as dur

RANDOM_SEED = 42
N_SPLITS = 5
N_REPEATS = 5
GBM_SEEDS = [0, 1, 2]

IDRE = re.compile(r"([A-Za-z0-9_-]{11})")
FEATURE_META = {"parkceleb_group", "parkceleb_label", "speaker_id", "video_id",
                "diarization_speaker", "source_type", "audio_file", "audio_ext",
                "audio_path", "intensity_ok", "intensity_error", "pause_ok",
                "pause_error", "syllable_ok", "syllable_error", "tot_name", "age"}

def _mean_std(a):
    return f"{np.mean(a):.4f} ± {np.std(a):.4f}"

def _mc_impute(X):
    X = np.asarray(X, dtype=float)
    med = np.nanmedian(X, axis=0)
    return np.where(np.isnan(X), np.where(np.isnan(med), 0.0, med), X)

def load_nv_age():
    hc = pd.read_csv(f"{DATA}/neurovoz/metadata/metadata_hc.csv")
    pd_ = pd.read_csv(f"{DATA}/neurovoz/metadata/metadata_pd.csv")
    mt = pd.concat([hc, pd_], ignore_index=True)
    mt["_pid"] = mt["ID"].map(lambda i: f"{int(i):04d}")
    mt["Age"] = pd.to_numeric(mt["Age"], errors="coerce")
    age_by_pid = mt.groupby("_pid")["Age"].median()
    base = f"{DATA}/neurovoz/extracted_features"
    intens = pd.read_csv(f"{base}/intensity_features_neurovoz.csv")
    pause = pd.read_csv(f"{base}/pause_features_neurovoz.csv")
    syll = pd.read_csv(f"{base}/syllable_features_neurovoz.csv")
    meta_cols = {"Unnamed: 0", "tot_name"}
    ki, kp, ks = (intens["tot_name"].apply(dur._stem),
                  pause["tot_name"].apply(dur._stem),
                  syll["tot_name"].apply(dur._stem))
    common = sorted(set(ki) & set(kp) & set(ks))
    X = pd.concat([dur._sl(intens, ki, common, meta_cols),
                   dur._sl(pause, kp, common, meta_cols),
                   dur._sl(syll, ks, common, meta_cols)], axis=1).values.astype(np.float64)
    Y = np.array([1 if s.startswith("PD_") else 0 for s in common], dtype=np.float64)
    age = np.array([age_by_pid.get(s[-4:], np.nan) for s in common], dtype=np.float64)
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(age))
    return X[ok], Y[ok], age[ok]

def load_pc_age():
    base = f"{DATA}/parkceleb_extracted"
    intens = pd.read_csv(f"{base}/intensity_features_parkceleb.csv")
    pause = pd.read_csv(f"{base}/pause_features_parkceleb.csv")
    syll = pd.read_csv(f"{base}/syllable_features_parkceleb.csv")
    ages = pd.read_csv(f"{DATA}/parkceleb_ages.csv")
    akey = ages["audio_file"].map(lambda f: IDRE.search(f).group(1))
    ages = ages.assign(_key=akey).drop_duplicates("_key").set_index("_key")
    kid = intens["audio_file"].map(
        lambda f: (lambda m: m.group(1) if m else np.nan)(IDRE.search(str(f))))
    m = kid.isin(ages.index)
    intens, pause, syll, kid = intens[m], pause[m], syll[m], kid[m]
    X = pd.concat([intens.loc[:, ~intens.columns.isin(FEATURE_META)],
                   pause.loc[:, ~pause.columns.isin(FEATURE_META)],
                   syll.loc[:, ~syll.columns.isin(FEATURE_META)]], axis=1)
    age = kid.map(ages["age"]).values.astype(np.float64)
    Y = intens["parkceleb_label"].values.astype(np.float64)
    sid = intens["speaker_id"].astype(str).values
    return X.values.astype(np.float64), Y, age, sid

def _age_weights(a_src, a_trg):
    rng = np.random.RandomState(0)
    jit = lambda a: a + 0 if np.std(a) > 1e-9 else rng.normal(0.0, 0.25, len(a))
    k_src, k_trg = gaussian_kde(jit(a_src)), gaussian_kde(jit(a_trg))
    w = np.exp(k_trg.logpdf(a_src) - k_src.logpdf(a_src))
    lo = max(np.percentile(w, 1), 0.01)
    hi = min(np.percentile(w, 99), 10.0)
    return np.clip(w, lo, hi) / np.mean(np.clip(w, lo, hi))

def _align_src_to_trg(X_src, X_trg):
    m_s, s_s = np.nanmean(X_src, 0), np.nanstd(X_src, 0) + 1e-12
    m_t, s_t = np.nanmean(X_trg, 0), np.nanstd(X_trg, 0) + 1e-12
    return (X_src - m_s) / s_s * s_t + m_t

def _gbm(seed=0):
    return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, random_state=seed)

def _domain_gap(X_src, X_trg):
    m_s, m_t = np.nanmean(X_src, 0), np.nanmean(X_trg, 0)
    s_s = np.nanstd(X_src, 0) + 1e-12
    return (m_t - m_s) / s_s

def main():
    print("Stage 2 (U=age, RAW features): causal transport NV -> PC")
    X_nv, y_nv, age_nv = load_nv_age()
    X_pc, y_pc, age_pc, sid_pc = load_pc_age()
    X_nv, X_pc = _mc_impute(X_nv), _mc_impute(X_pc)

    print(f"  NV: {len(y_nv)} recordings (age {np.nanmin(age_nv):.0f}-{np.nanmax(age_nv):.0f}), "
          f"PC: {len(y_pc)} clips, {len(np.unique(sid_pc))} speakers "
          f"(age {age_pc.min():.0f}-{age_pc.max():.0f})")

    dmu = _domain_gap(X_nv, X_pc)
    print(f"  feature domain gap (raw): mean |dmu|/sigma = {np.mean(np.abs(dmu)):.3f}, "
          f"frac |dmu|>0.5sigma = {np.mean(np.abs(dmu) > 0.5):.2f}")

    w = _age_weights(age_nv, age_pc)
    ess = (w.sum() ** 2) / (w ** 2).sum()
    print(f"  sID age weights (NV -> PC): ESS = {ess:.0f}/{len(age_nv)} "
          f"({ess/len(age_nv):.2f}), w in [{w.min():.3f}, {w.max():.3f}]")

    nv_U = np.column_stack([X_nv, age_nv])
    pc_U = np.column_stack([X_pc, age_pc])
    X_nv_al = _align_src_to_trg(X_nv, X_pc)
    nv_al_U = np.column_stack([X_nv_al, age_nv])

    rows = {"transported": [], "transported_p": [], "weighted_p": [],
            "aligned_p": [], "localized_p": [], "placebo_p": [], "no_u_p": [],
            "age_only_p": [], "age_only_auc": []}
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=RANDOM_SEED)
    for fold, (pc_tr, pc_te) in enumerate(rskf.split(pc_U, y_pc)):
        for seed in GBM_SEEDS:
            m_nv = _gbm(seed).fit(nv_U, y_nv.ravel())
            p = m_nv.predict_proba(pc_U[pc_te])[:, 1]
            rows["transported"].append(roc_auc_score(y_pc[pc_te], p))
            rows["transported_p"].append(dur.patient_level_metrics(
                y_pc[pc_te], p, sid_pc[pc_te]))

            m_w = _gbm(seed).fit(nv_U, y_nv.ravel(), sample_weight=w)
            rows["weighted_p"].append(dur.patient_level_metrics(
                y_pc[pc_te], m_w.predict_proba(pc_U[pc_te])[:, 1], sid_pc[pc_te]))

            m_a = _gbm(seed).fit(nv_al_U, y_nv.ravel())
            rows["aligned_p"].append(dur.patient_level_metrics(
                y_pc[pc_te], m_a.predict_proba(pc_U[pc_te])[:, 1], sid_pc[pc_te]))

            m_au = _gbm(seed).fit(age_nv.reshape(-1, 1), y_nv.ravel())
            p_au = m_au.predict_proba(age_pc[pc_te].reshape(-1, 1))[:, 1]
            rows["age_only_auc"].append(roc_auc_score(y_pc[pc_te], p_au))
            rows["age_only_p"].append(dur.patient_level_metrics(
                y_pc[pc_te], p_au, sid_pc[pc_te]))

            rng = np.random.RandomState(RANDOM_SEED + fold * 10 + seed)
            pc_placebo = np.column_stack([X_pc, rng.permutation(age_pc)])
            rows["placebo_p"].append(dur.patient_level_metrics(
                y_pc[pc_te], m_nv.predict_proba(pc_placebo[pc_te])[:, 1], sid_pc[pc_te]))

            rows["no_u_p"].append(dur.patient_level_metrics(
                y_pc[pc_te],
                _gbm(seed).fit(X_nv, y_nv.ravel()).predict_proba(X_pc[pc_te])[:, 1],
                sid_pc[pc_te]))

            m_pc = _gbm(seed).fit(pc_U[pc_tr], y_pc[pc_tr].ravel())
            rows["localized_p"].append(dur.patient_level_metrics(
                y_pc[pc_te], m_pc.predict_proba(pc_U[pc_te])[:, 1], sid_pc[pc_te]))

    n = len(rows["transported"])
    print(f"\n  transported (AUC / patient BAS): {_mean_std(rows['transported'])} / "
          f"{_mean_std(rows['transported_p'])}")
    print(f"  sID-weighted    (patient BAS):   {_mean_std(rows['weighted_p'])}")
    print(f"  feature-aligned (patient BAS):   {_mean_std(rows['aligned_p'])}")
    print(f"  PC-localized    (patient BAS):   {_mean_std(rows['localized_p'])}")
    print(f"  placebo         (patient BAS):   {_mean_std(rows['placebo_p'])}")
    print(f"  no-U            (patient BAS):   {_mean_std(rows['no_u_p'])}")
    print(f"  AGE-ONLY  (AUC / patient BAS):   {_mean_std(rows['age_only_auc'])} / "
          f"{_mean_std(rows['age_only_p'])}  <- must be ~chance")
    print(f"  ({N_SPLITS}-fold x {N_REPEATS} repeats x {len(GBM_SEEDS)} GBM seeds = "
          f"{n} fits per variant)\n")

    out = (f"Stage 2 U=age raw features (NV->PC), {n} fits/variant\n"
           f"transported AUC/BAS: {_mean_std(rows['transported'])} / {_mean_std(rows['transported_p'])}\n"
           f"weighted BAS: {_mean_std(rows['weighted_p'])}\n"
           f"aligned BAS: {_mean_std(rows['aligned_p'])}\n"
           f"localized BAS: {_mean_std(rows['localized_p'])}\n"
           f"placebo BAS: {_mean_std(rows['placebo_p'])}\n"
           f"no_u BAS: {_mean_std(rows['no_u_p'])}\n"
           f"age_only AUC/BAS: {_mean_std(rows['age_only_auc'])} / {_mean_std(rows['age_only_p'])}\n")
    with open(f"{BASE}/notes/transport_u_age_raw.txt", "w") as f:
        f.write(out)
    print("saved notes/transport_u_age_raw.txt")

if __name__ == "__main__":
    main()
