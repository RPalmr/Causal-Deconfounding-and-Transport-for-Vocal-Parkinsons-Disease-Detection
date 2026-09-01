"""grbas rating cleaner for neurovoz in the dissertation pipeline.

reads the raw per-task grbas csvs in data/grbas/, normalises subscore
columns g/r/b/a/s and total, joins each task against the audio roster
under data/neurovoz/audios by normalised task stem, and optionally
back-fills rows for unrated audios; writes one cleaned csv per task."""

import argparse
import os
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
AUDIO_DIR = BASE / "data" / "neurovoz" / "audios"
GRBAS_DIR = BASE / "data" / "grbas"

SUBSCORES = ["G", "R", "B", "A", "S"]
AUDIO_RE = re.compile(r"^(?:PD|HC)_(.+)_(\d{4})\.wav$")
PID_RE = re.compile(r"(\d{4})$")


def norm(s):
    return str(s).lower().replace("_", "")


def audio_roster(audio_dir):
    roster = {}
    for audio in sorted(os.listdir(audio_dir)):
        m = AUDIO_RE.match(str(audio))
        if not m:
            continue
        roster.setdefault(norm(m.group(1)), []).append(f"{m.group(1)}_{m.group(2)}.wav")
    return roster


def build_grbas(grbas_dir, audio_dir, include_extra):
    roster = audio_roster(audio_dir)
    tasks = {}
    for f in sorted(os.listdir(grbas_dir)):
        if not f.endswith(".csv"):
            continue
        stem = os.path.splitext(f)[0]
        df = pd.read_csv(grbas_dir / f)
        id_col = "text_patient_id" if "text_patient_id" in df.columns else df.columns[0]
        rows = []
        for _, r in df.iterrows():
            gid = os.path.splitext(os.path.basename(str(r[id_col])))[0].strip()
            row = {"text_patient_id": f"{gid}.wav"}
            for k in SUBSCORES:
                row[k] = (pd.to_numeric(r[k], errors="coerce")
                          if k in df.columns else pd.NA)
            row["TOTAL"] = (pd.to_numeric(r["TOTAL"], errors="coerce")
                            if "TOTAL" in df.columns else pd.NA)
            row["COMMENTS"] = (str(r["COMMENTS"]).strip()
                               if "COMMENTS" in df.columns
                               and pd.notna(r["COMMENTS"]) else "")
            rows.append(row)
        out = pd.DataFrame(rows)
        sub = out[SUBSCORES].apply(pd.to_numeric, errors="coerce")
        tot = out["TOTAL"]
        out["TOTAL"] = tot.where(tot.notna(), sub.sum(axis=1, min_count=1))

        audios = roster.get(norm(stem), [])
        rated = set(out["text_patient_id"])
        missing_in_audio = sorted(rated - set(audios))
        if missing_in_audio:
            print(f"  {stem}: {len(missing_in_audio)} rated rows have no audio file")
        if include_extra:
            extra = [a for a in audios if a not in rated]
            if extra:
                rows = []
                for a in extra:
                    rows.append({"text_patient_id": a,
                                 **{k: pd.NA for k in SUBSCORES},
                                 "TOTAL": pd.NA, "COMMENTS": ""})
                out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)
        tasks[stem] = out
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(GRBAS_DIR))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--include-extra", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = build_grbas(GRBAS_DIR, AUDIO_DIR, args.include_extra)
    for stem, df in tasks.items():
        dst = out / f"{stem}.csv"
        if dst.exists() and not args.overwrite:
            print(f"skip {dst} (exists, --overwrite to replace)")
            continue
        df.to_csv(dst, index=False)
        print(f"wrote {dst} ({len(df)} rows)")


if __name__ == "__main__":
    main()