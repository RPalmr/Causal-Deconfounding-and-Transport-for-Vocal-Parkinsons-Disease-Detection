"""
Transportable Causal Bootstrapping with U = disease duration (NeuroVoz -> ParkCeleb).

    Duration -> PD;  Duration -> Voice;  PD -> Voice;
    Pause    -> Voice;  S_dur -> Duration;  S_pause -> Pause

U (source): "Time Disease (years)" from metadata_pd.csv (0 for HC).
U (target): years_from_diagnosis x sign(before/after) from speakers_info.csv (0 for CN).

    w ~ P*(Duration, Pause) / P(Duration, Pause | PD)
"""

import sys, os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{BASE}/data"
for p in [f"{BASE}/libraries/GRAPL", f"{BASE}/libraries/CausalBootstrapping",
          f"{BASE}/tCB", BASE]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sID_expr import sid
import grapl.dsl as dsl
from causalBootstrapping import bootstrapper

TEST_SIZE = 0.2
RANDOM_SEED = 42
N_RUNS = 500
N_TILE = 10


def _stem(x):
    s = os.path.basename(str(x))
    while s.endswith(".wav"):
        s = s[:-4]
    return s.replace("audios-", "").replace("audios/", "")


def _nv_duration_by_patient():
    hc = pd.read_csv(f"{DATA}/neurovoz/metadata/metadata_hc.csv")
    pd_ = pd.read_csv(f"{DATA}/neurovoz/metadata/metadata_pd.csv")
    mt = pd.concat([hc, pd_], ignore_index=True)
    mt["_pid"] = mt["ID"].map(lambda i: f"{int(i):04d}")
    mt["Time Disease (years)"] = pd.to_numeric(mt["Time Disease (years)"], errors="coerce")
    dur = mt.groupby("_pid")["Time Disease (years)"].median()
    return dur.where(dur.notna(), 0.0).to_dict()


def _sl(df, kv, common, meta_cols):
    return (df.loc[kv.isin(common)].loc[:, ~df.columns.isin(meta_cols)]
            .set_index(kv[kv.isin(common)].values))


def load_neurovoz():
    base = f"{DATA}/neurovoz/extracted_features"
    intens = pd.read_csv(f"{base}/intensity_features_neurovoz.csv")
    pause = pd.read_csv(f"{base}/pause_features_neurovoz.csv")
    syll = pd.read_csv(f"{base}/syllable_features_neurovoz.csv")
    meta_cols = {"Unnamed: 0", "tot_name"}
    ki, kp, ks = intens["tot_name"].apply(_stem), pause["tot_name"].apply(_stem), syll["tot_name"].apply(_stem)
    common = sorted(set(ki) & set(kp) & set(ks))
    X = pd.concat([_sl(intens, ki, common, meta_cols), _sl(pause, kp, common, meta_cols),
                   _sl(syll, ks, common, meta_cols)], axis=1).values.astype(np.float64)
    Y = np.array([1 if s.startswith("PD_") else 0 for s in common], dtype=np.float64)
    dur = _nv_duration_by_patient()
    U = np.array([dur.get(s[-4:], np.nan) for s in common], dtype=np.float64)
    P = np.nanmedian(_sl(pause, kp, common, meta_cols).select_dtypes(include=[np.number]).values, axis=1)
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(U))
    return X[ok], Y[ok], U[ok], P[ok]


def load_parkceleb():
    base = f"{DATA}/parkceleb_extracted"
    intens = pd.read_csv(f"{base}/intensity_features_parkceleb.csv")
    pause = pd.read_csv(f"{base}/pause_features_parkceleb.csv")
    syll = pd.read_csv(f"{base}/syllable_features_parkceleb.csv")
    meta = {"parkceleb_group", "parkceleb_label", "speaker_id", "video_id",
            "diarization_speaker", "source_type", "audio_file", "audio_ext", "audio_path",
            "intensity_ok", "intensity_error", "pause_ok", "pause_error",
            "syllable_ok", "syllable_error", "tot_name"}
    X = pd.concat([intens.loc[:, ~intens.columns.isin(meta)],
                   pause.loc[:, ~pause.columns.isin(meta)],
                   syll.loc[:, ~syll.columns.isin(meta)]], axis=1).values.astype(np.float64)

    vid_dur = {}
    for group in ("CN", "PD"):
        gp = f"{DATA}/ParkCeleb/{group}"
        if not os.path.isdir(gp):
            continue
        for pdir in os.listdir(gp):
            ppath = f"{gp}/{pdir}"
            if not os.path.isdir(ppath):
                continue
            for vdir in os.listdir(ppath):
                si = f"{ppath}/{vdir}/speakers_info.csv"
                if not os.path.isfile(si):
                    continue
                try:
                    info = pd.read_csv(si)
                    tgt = info[info["status"] == "target"]
                    if tgt.empty:
                        continue
                    row = tgt.iloc[0]
                    yfd = str(row["years_from_diagnosis"]).strip()
                    if group == "CN" or yfd in ("", "same", "nan"):
                        vid_dur[vdir] = 0.0
                    else:
                        sign = 1 if str(row["before_after_diagnosis"]).strip().lower() == "after" else -1
                        vid_dur[vdir] = float(sign * float(yfd))
                except Exception:
                    vid_dur[vdir] = 0.0

    U = np.array([vid_dur.get(v, np.nan) for v in intens["video_id"].astype(str).values], dtype=np.float64)
    Y = intens["parkceleb_label"].values.astype(np.float64)
    P = np.nanmedian(pause.loc[:, ~pause.columns.isin(meta)].select_dtypes(include=[np.number]).values, axis=1)
    sid = intens["speaker_id"].astype(str).values
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(U))
    return X[ok], Y[ok], U[ok], P[ok], sid[ok]


def build_causal_graph():
    return dsl.GraplDSL().readgrapl(
        "Duration; PD; Voice; Pause; S_dur; S_pause;"
        "Duration -> PD; Duration -> Voice; PD -> Voice; Pause -> Voice;"
        "S_dur -> Duration; S_pause -> Pause;"
    )


def compute_transport_weights(y_src, dur_src, pause_src, dur_trg, pause_trg):
    n = len(y_src)
    nv_dp = np.column_stack([dur_src.ravel(), pause_src.ravel()])
    kde_dp_given_pd = {}
    for pd_val in (0, 1):
        dp = nv_dp[y_src.ravel() == pd_val]
        if dp.shape[0] > 3:
            if np.var(dp[:, 0]) < 1e-9:
                rng = np.random.RandomState(0)
                dp = dp.copy()
                dp[:, 0] += rng.normal(0.0, 0.25, dp.shape[0])
            kde_dp_given_pd[pd_val] = gaussian_kde(dp.T)
        else:
            kde_dp_given_pd[pd_val] = None
    kde_target_dur = gaussian_kde(dur_trg)
    kde_target_pause = gaussian_kde(pause_trg)
    log_floor = np.log(1e-12)
    log_target = np.maximum(
        kde_target_dur.logpdf(dur_src.reshape(1, -1))
        + kde_target_pause.logpdf(pause_src.reshape(1, -1)),
        log_floor,
    )
    weights = np.zeros((n, 2))
    for intv_pd in (0, 1):
        k = kde_dp_given_pd.get(intv_pd)
        log_source = np.full(n, log_floor) if k is None else k.logpdf(nv_dp.T)
        weights[:, intv_pd] = np.exp(log_target - np.maximum(log_source, log_floor)) + 1e-10
    return weights


def balanced_acc(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)
    tp, fn = ((y_pred == 1) & (y_true == 1)).sum(), ((y_pred == 0) & (y_true == 1)).sum()
    tn, fp = ((y_pred == 0) & (y_true == 0)).sum(), ((y_pred == 1) & (y_true == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (sens + spec)


def patient_level_metrics(y, proba, patient_ids):
    pdf = pd.DataFrame({"y": y.ravel(), "proba": proba.ravel(),
                        "patient": np.asarray(patient_ids)}).groupby("patient").mean()
    return balanced_acc(pdf["y"].values, pdf["proba"].values)


def fit_logit(X, y):
    y = np.asarray(y).ravel()
    X = np.asarray(X, dtype=float)
    med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isnan(med), 0.0, med)
    X = np.where(np.isnan(X), col_med, X)
    keep = np.var(X, axis=0) > 0
    return {"model": LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                        max_iter=1000, random_state=0).fit(X[:, keep], y),
            "col_med": col_med[keep], "keep": keep, "n_full": len(keep)}


def predict_logit(model, X):
    X = np.asarray(X, dtype=float)
    if X.shape[1] != model["n_full"] or not model["keep"].all():
        X = X[:, model["keep"]]
    return model["model"].predict_proba(np.where(np.isnan(X), model["col_med"], X))[:, 1]


def main():
    G = build_causal_graph()
    result = sid(Y={"Voice"}, X={"PD"}, S={"S_dur", "S_pause"}, D=G)
    if result is None:
        print("WARNING: effect NOT transportable with this graph.")
        return
    print("sID formula RHS:", result.rhs.tocondstr())

    X_nv, y_nv, dur_nv, pause_nv = load_neurovoz()
    X_pc, y_pc, dur_pc, pause_pc, sid_pc = load_parkceleb()
    print(f"NV: {len(y_nv)} recordings, PC: {len(y_pc)} clips, {len(np.unique(sid_pc))} speakers")

    pca_ssl = PCA(n_components=1)
    voice_nv = pca_ssl.fit_transform(StandardScaler().fit_transform(np.nan_to_num(X_nv, nan=0.0))).ravel()
    voice_pc = pca_ssl.transform(np.nan_to_num(X_pc, nan=0.0)).ravel()

    weights = compute_transport_weights(y_nv, dur_nv, pause_nv, dur_pc, pause_pc)
    ess = (weights.sum() ** 2) / (weights ** 2).sum()
    print(f"KDE weights: ESS = {ess:.0f} / {len(y_nv)} ({ess/len(y_nv):.2f})")

    pc_tr, pc_te = train_test_split(np.arange(len(y_pc)), test_size=TEST_SIZE,
                                    random_state=RANDOM_SEED, stratify=y_pc)
    scaler_feat = StandardScaler()
    voice_train_scaled = scaler_feat.fit_transform(voice_pc[pc_tr].reshape(-1, 1))
    voice_test_scaled = scaler_feat.transform(voice_pc[pc_te].reshape(-1, 1))

    conf_model = fit_logit(voice_train_scaled, y_pc[pc_tr])
    conf_ba = patient_level_metrics(y_pc[pc_te], predict_logit(conf_model, voice_test_scaled), sid_pc[pc_te])

    nv_data_big = {"Voice": np.tile(voice_nv.reshape(-1, 1), (N_TILE, 1)),
                   "PD": np.tile(y_nv.reshape(-1, 1), (N_TILE, 1))}
    weights_big = np.tile(weights, (N_TILE, 1))
    tcb_bas_list = []
    for run in range(N_RUNS):
        bd, _ = bootstrapper(nv_data_big, weights_big, intv_var_name_in_data=["PD"],
                             mode="fast", random_state=RANDOM_SEED + run)
        X_boot = bd["Voice"].astype(np.float64)
        y_boot = bd["intv_PD"].astype(np.int32).ravel()
        m = fit_logit(StandardScaler().fit_transform(X_boot), y_boot)
        tcb_bas_list.append(patient_level_metrics(y_pc[pc_te], predict_logit(m, voice_test_scaled), sid_pc[pc_te]))

    tcb_bas, tcb_std = np.mean(tcb_bas_list), np.std(tcb_bas_list)
    print(f"PC confounded:   BAS = {conf_ba:.4f}")
    print(f"TCB transported: BAS = {tcb_bas:.4f} ± {tcb_std:.4f}")


if __name__ == "__main__":
    main()
