"""neurovoz audio resampler for the dissertation pipeline.

loads every wav under data/neurovoz/audios, force-mono mixes stereo clips,
resamples any rate outside {8k, 16k, 32k, 48k} to 32 khz pcm_16 via
scipy resample_poly, and writes the results to data/neurovoz/downsample_audios
together with a conversion_log.csv used by downstream feature extraction."""

from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm
import pandas as pd

ALLOWED_SR = {8000, 16000, 32000, 48000}

def convert_and_save(
    in_path,
    out_path,
    target_sr=16000,
    allowed_sr=ALLOWED_SR,
    force_mono=True,
    out_subtype="PCM_16",
):
    in_path = Path(in_path)
    out_path = Path(out_path)

    y, sr = sf.read(str(in_path), always_2d=True)

    if force_mono and y.shape[1] > 1:
        y = y.mean(axis=1, keepdims=False)
    else:
        y = y[:, 0]

    y = y.astype(np.float32)

    new_sr = sr
    did_resample = False
    if sr not in allowed_sr:
        y = resample_poly(y, target_sr, sr).astype(np.float32)
        new_sr = target_sr
        did_resample = True

    y = np.clip(y, -1.0, 1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, new_sr, subtype=out_subtype)

    return {
        "input": str(in_path),
        "output": str(out_path),
        "sr_in": sr,
        "sr_out": new_sr,
        "resampled": did_resample,
        "status": "ok",
        "error": "",
    }

neurovoz_dir = Path("/Volumes/ESSD/transportable causal bootstrapping/data/neurovoz")
in_dir = neurovoz_dir / "audios"
out_dir = neurovoz_dir / "downsample_audios"
root_dir = None

files = sorted(in_dir.glob("*.wav"))

logs = []
for f in tqdm(files, total=len(files)):
    f = Path(f)

    if root_dir is not None:
        rel = f.relative_to(root_dir)
        out_path = out_dir / rel.with_suffix(".wav")
    else:
        out_path = out_dir / (f.stem + ".wav")

    try:
        logs.append(convert_and_save(f, out_path, target_sr=32000))
    except Exception as e:
        logs.append({
            "input": str(f),
            "output": str(out_path),
            "sr_in": None,
            "sr_out": None,
            "resampled": None,
            "status": "fail",
            "error": repr(e),
        })

df_log = pd.DataFrame(logs)
df_log.to_csv(out_dir / "conversion_log.csv", index=False, encoding="utf-8-sig")

print("Done.")
print(df_log["status"].value_counts())
print("Saved to:", out_dir)