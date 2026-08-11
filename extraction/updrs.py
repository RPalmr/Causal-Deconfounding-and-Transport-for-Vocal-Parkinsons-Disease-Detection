import argparse
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
METADATA_PD = BASE / "data" / "neurovoz" / "metadata" / "metadata_pd.csv"
METADATA_HC = BASE / "data" / "neurovoz" / "metadata" / "metadata_hc.csv"


def build_updrs(meta_pd, meta_hc):
    hc = pd.read_csv(meta_hc)
    pd_ = pd.read_csv(meta_pd)
    mt = pd.concat([hc, pd_], ignore_index=True)
    mt["_pid"] = mt["ID"].map(lambda i: f"{int(i):04d}")
    mt["UPDRS scale"] = pd.to_numeric(mt["UPDRS scale"], errors="coerce")
    out = pd.DataFrame({"pid": sorted(mt["_pid"].unique()), "updrs": 0.0})
    out["updrs"] = out["pid"].map(mt.groupby("_pid")["UPDRS scale"].median())
    out["group"] = out["pid"].map(mt.groupby("_pid")["Group"].first())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "data" / "updrs" / "updrs.csv"))
    args = ap.parse_args()
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"skip {dst} (exists, --overwrite to replace)")
        return
    updrs = build_updrs(METADATA_PD, METADATA_HC)
    updrs.to_csv(dst, index=False)
    print(f"wrote {dst} ({len(updrs)} patients, "
          f"{int(updrs['updrs'].notna().sum())} with UPDRS)")


if __name__ == "__main__":
    main()