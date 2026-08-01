import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT          = Path(__file__).resolve().parents[2]
NV_FEAT_PATH  = ROOT / "data/neurovoz/extracted_features"
NV_META_PATH  = ROOT / "data/neurovoz/metadata"
NV_GRBAS_PATH = ROOT / "data/grbas"
NV_SSL_PATH   = ROOT / "data/neurovoz/ssl_features/embeddings.feather"
DYS_PATH      = ROOT / "paper/dysphonia_features_neurovoz.csv"
EXCLUDE_IDS   = {"0056", "0074", "0121", "0135"}
TEST_SIZE     = 0.2
RANDOM_SEED   = 42
GRBAS_SUBSCORE_COLS = ("G", "R", "B", "A", "S")
GRBAS_VOWEL_TASKS = (
    "A1", "A2", "A3", "E1", "E2", "E3", "I1", "I2", "I3",
    "O1", "O2", "O3", "U1", "U2", "U3",
)
VOWEL_LETTERS = ("A", "E", "I", "O", "U")
BAD_COLS = [
    "int_stddev_pitch",
    "pau_Total_Time_VADInt_1", "pau_Total_Time_VADInt_2", "pau_Total_Time_VADInt_3",
    "syl_nsyll", "syl_npause", "syl_dur(s)",
]

def _load_nv_raw() -> pd.DataFrame:
    int_df = pd.read_csv(NV_FEAT_PATH / "intensity_features_neurovoz.csv", index_col=0)
    pau_df = pd.read_csv(NV_FEAT_PATH / "pause_features_neurovoz.csv", index_col=0)
    syl_df = pd.read_csv(NV_FEAT_PATH / "syllable_features_neurovoz.csv", index_col=0)
    def _parse(s):
        base = s.split("/")[-1]
        return base[-8:-4], base[3:-9]
    idx = pd.MultiIndex.from_tuples(int_df["tot_name"].apply(_parse), names=["patient", "test"])
    int_f = int_df.drop(columns=["tot_name"]).add_prefix("int_")
    pau_f = pau_df.drop(columns=[c for c in pau_df.columns if "tot_name" in c]).add_prefix("pau_")
    syl_f = syl_df.drop(columns=["tot_name"]).add_prefix("syl_")
    X = pd.concat([int_f, pau_f, syl_f], axis=1)
    X.index = idx
    return X.drop(columns=[c for c in BAD_COLS if c in X.columns])

def _load_meta() -> pd.DataFrame:
    meta = pd.concat([
        pd.read_csv(NV_META_PATH / "metadata_hc.csv"),
        pd.read_csv(NV_META_PATH / "metadata_pd.csv"),
    ])
    meta.index = meta["ID"].map(lambda i: f"{i:04d}")
    meta = meta[~meta.index.duplicated(keep="first")]
    meta["PD"] = meta["Group"] == "PD"
    return meta

def _parse_grbas_id(filename: str) -> str | None:
    m = re.search(r"(\d{4})\.wav$", str(filename).strip())
    return m.group(1) if m else None

def _load_grbas() -> pd.Series:
    frames = []
    for task in GRBAS_VOWEL_TASKS:
        path = NV_GRBAS_PATH / f"{task}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        id_col = "text_patient_id" if "text_patient_id" in df.columns else df.columns[0]
        df["patient"] = df[id_col].map(_parse_grbas_id)
        sub_cols = [c for c in GRBAS_SUBSCORE_COLS if c in df.columns]
        if not sub_cols:
            continue
        sub = df[sub_cols].apply(pd.to_numeric, errors="coerce")
        df["grbas_total"] = sub.sum(axis=1, skipna=False)
        frames.append(df[["patient", "grbas_total"]].dropna())
    grbas = pd.concat(frames, ignore_index=True)
    return grbas.groupby("patient")["grbas_total"].median()

def _vowel_letter(test_name: str) -> str | None:
    for v in VOWEL_LETTERS:
        if test_name.startswith(v):
            return v
    return None

def _load_ssl() -> pd.DataFrame:
    return pd.read_feather(NV_SSL_PATH)

def _load_dys() -> pd.DataFrame:
    df = pd.read_csv(DYS_PATH, dtype={"patient": str})
    df.set_index(["patient", "test"], inplace=True)
    return df

def load_neurovoz_joint():
    meta = _load_meta()
    grbas = _load_grbas()
    ssl = _load_ssl()
    dys = _load_dys()
    X_all = _load_nv_raw()
    vowel_tasks = [t for t in X_all.index.get_level_values("test").unique()
                   if _vowel_letter(t) is not None]
    X_vowel = X_all.loc[X_all.index.get_level_values("test").isin(vowel_tasks)]
    X_vowel = X_vowel[~X_vowel.index.get_level_values("patient").isin(EXCLUDE_IDS)]
    common_idx = X_vowel.index.intersection(dys.index)
    X_vowel = X_vowel.loc[common_idx].join(dys.loc[common_idx], how="left")
    ssl_cols = ssl.columns.tolist()
    common_idx2 = X_vowel.index.intersection(ssl.index)
    X_vowel = X_vowel.loc[common_idx2].join(ssl.loc[common_idx2], how="left")
    handcrafted_names = [c for c in X_vowel.columns if not c.startswith("ssl_")]
    ssl_names = [c for c in X_vowel.columns if c.startswith("ssl_")]
    n_hand = len(handcrafted_names)
    patient_rows = {}
    for pid, pdf in X_vowel.groupby(level="patient"):
        if pid not in meta.index:
            continue
        if pid not in grbas.index:
            continue
        vowel_vecs = {}
        for test_name, vdf in pdf.groupby("test"):
            vletter = _vowel_letter(test_name)
            if vletter is None:
                continue
            hc_vals = vdf[handcrafted_names].fillna(0).values
            if len(hc_vals) == 0:
                continue
            vowel_vecs.setdefault(vletter, []).append(np.nanmedian(hc_vals, axis=0))
        hc_row = []
        for vletter in VOWEL_LETTERS:
            reps = vowel_vecs.get(vletter)
            if reps:
                hc_row.append(np.nanmedian(reps, axis=0))
            else:
                hc_row.append(np.full(n_hand, np.nan))
        hc_features = np.concatenate(hc_row)
        ssl_vals = pdf[ssl_names].fillna(0).values
        if len(ssl_vals) > 0:
            ssl_median = np.nanmedian(ssl_vals, axis=0)
        else:
            ssl_median = np.full(len(ssl_names), np.nan)
        patient_rows[pid] = np.concatenate([hc_features, ssl_median])
    patients_sorted = sorted(patient_rows.keys())
    X = np.array([patient_rows[p] for p in patients_sorted])
    m = meta.loc[patients_sorted]
    y = m["PD"].astype(int).values.ravel()
    return X, y, np.arange(len(y))

def balanced_acc(y_true, proba, threshold=0.5) -> float:
    y_pred = (proba >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (sens + spec)

def fit_logit(X, y):
    return LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                              max_iter=1000, random_state=0).fit(X, y)

def predict_logit(model, X):
    return model.predict_proba(X)[:, 1]

def main():
    X_nv, y_nv, pat_nv = load_neurovoz_joint()
    print(f"NV: {X_nv.shape[0]} patients, {X_nv.shape[1]} features "
          f"(PD={y_nv.sum()}, CN={len(y_nv)-y_nv.sum()})")

    tr, te = train_test_split(np.arange(len(y_nv)), test_size=TEST_SIZE,
                               random_state=RANDOM_SEED, stratify=y_nv)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_nv[tr])
    X_te = scaler.transform(X_nv[te])
    model = fit_logit(X_tr, y_nv[tr])
    p = predict_logit(model, X_te)
    ba = balanced_acc(y_nv[te], p)
    print(f"  NV confounded: BAS={ba:.4f}")

if __name__ == "__main__":
    main()
