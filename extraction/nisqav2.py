"""nisqa speech-quality feature extractor for parkceleb in the dissertation.

resolves each row of the parkceleb extraction manifest to a local audio
file, cuts a centred 30 s 48 khz mono snippet with ffmpeg, runs the nisqa
pretrained model (mos/noi/dis/col/loud), merges the predictions back to
the manifest, and writes them incrementally to
data/parkceleb_extracted/quality_features_parkceleb.csv (resumable)."""

import argparse, os, shutil, subprocess, sys, tempfile
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = BASE + '/data/parkceleb_extracted/extraction_manifest_parkceleb.csv'
OUT_CSV = BASE + '/data/parkceleb_extracted/quality_features_parkceleb.csv'
NISQA_DIR = BASE + '/deps/NISQA'
SNIPPET_SECS = 30.0
CHUNK = 75

def resolve(row):
    g, spk, vid = row.parkceleb_group, row.speaker_id, str(row.video_id)
    wav = '%s/data/ParkCeleb/%s/%s/%s/%s.wav' % (BASE, g, spk, vid, vid)
    if os.path.exists(wav):
        return wav
    for ext in ("opus", "m4a", "mp4", "webm", "ogg", "aac", "flac", "mov"):
        p = '%s/data/ParkCeleb/%s/%s/%s/%s.%s' % (BASE, g, spk, vid, vid, ext)
        if os.path.exists(p):
            return p
    return None

def ffmpeg_snippet(src, dur, dst):
    off = max(0.0, (dur - SNIPPET_SECS) / 2.0)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(off),
                    "-t", str(SNIPPET_SECS), "-i", src,
                    "-ar", "48000", "-ac", "1", dst], check=True)

def predict_dir(snippets, work):
    sys.path.insert(0, NISQA_DIR)
    import torch
    from nisqa.NISQA_model import nisqaModel
    args = dict(mode="predict_dir", pretrained_model=NISQA_DIR + "/weights/nisqa.tar",
                data_dir=snippets, output_dir=work, ms_channel=None,
                tr_bs_val=8, tr_num_workers=0, tr_parallel=False, ms_sr=48e3)
    d = nisqaModel(args)
    if torch.backends.mps.is_available():
        d.dev = torch.device("mps")
    d.model.to(d.dev)
    try:
        d.predict()
    except Exception:
        d.dev = torch.device("cpu")
        d.model.to(d.dev)
        d.predict()
    res = pd.read_csv(os.path.join(work, "NISQA_results.csv"))
    os.remove(os.path.join(work, "NISQA_results.csv"))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    m = pd.read_csv(MANIFEST)
    m["resolved"] = m.apply(resolve, axis=1)
    m = m[m["resolved"].notna()].reset_index(drop=True)
    q = pd.read_csv(OUT_CSV) if args.resume and os.path.exists(OUT_CSV) else \
        pd.DataFrame(columns=["audio_file", "speaker_id", "video_id",
                              "mos", "noi", "dis", "col", "loud"])
    done = set(q["video_id"].astype(str)) if len(q) else set()
    todo = m[~m["video_id"].astype(str).isin(done)].reset_index(drop=True)
    print("clips with audio: %d  already done: %d  remaining: %d" % (
        len(m), len(done), len(todo)))
    work = tempfile.mkdtemp(prefix="nisqa_")
    try:
        for i in range(0, len(todo), CHUNK):
            chunk = todo.iloc[i:i + CHUNK]
            snippets = os.path.join(work, "s")
            os.makedirs(snippets, exist_ok=True)
            for _, r in chunk.iterrows():
                ffmpeg_snippet(r["resolved"], float(r["duration_sec"]),
                               os.path.join(snippets, "%s.wav" % r["video_id"]))
            res = predict_dir(snippets, work)
            res["deg"] = res["deg"].astype(str).str.replace(r"\.wav$", "", regex=True)
            row = chunk.merge(res, left_on="video_id", right_on="deg",
                              how="inner")[["audio_file", "speaker_id",
                                            "video_id", "mos_pred",
                                            "noi_pred", "dis_pred",
                                            "col_pred", "loud_pred"]]
            row = row.rename(columns={"mos_pred": "mos", "noi_pred": "noi",
                                      "dis_pred": "dis", "col_pred": "col",
                                      "loud_pred": "loud"})
            q = pd.concat([q, row], ignore_index=True)
            q.to_csv(OUT_CSV, index=False)
            shutil.rmtree(snippets)
            print("  chunk %4d-%4d / %d  (cumulative %d clips)" % (
                i, min(i + CHUNK, len(todo)), len(todo), len(q)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("done: %d clips -> %s" % (len(q), OUT_CSV))
    print("unrecovered clips: %d" % int((~pd.read_csv(MANIFEST)["video_id"]
          .astype(str).isin(q["video_id"].astype(str))).sum()))

if __name__ == "__main__":
    main()