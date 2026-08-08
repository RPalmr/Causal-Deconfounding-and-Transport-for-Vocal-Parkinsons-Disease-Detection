import sys, os, glob, time
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

BASE = "/Volumes/ESSD/transportable causal bootstrapping"
MODEL = "facebook/wav2vec2-base"
LAYER = 3
SEG = 10.0

DEV = "mps" if torch.backends.mps.is_available() else "cpu"

def load_model():
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
    m = Wav2Vec2Model.from_pretrained(MODEL)
    m.to(DEV).eval()
    return fe, m

@torch.no_grad()
def embed_clip(fe, m, x, sr):
    n_seg = max(1, int(np.ceil(len(x) / (SEG * sr))))
    embs = np.zeros((n_seg, 768))
    for i in range(n_seg):
        a = int(i * SEG * sr); b = min(len(x), a + int(SEG * sr))
        seg = x[a:b]
        if b - a < int(SEG * sr):
            seg = np.pad(seg, (0, int(SEG * sr) - (b - a)))
        inp = fe(seg, sampling_rate=sr, return_tensors="pt").to(DEV)
        hs = m(**inp, output_hidden_states=True).hidden_states[LAYER]
        embs[i] = hs.squeeze().mean(0).cpu().numpy()
    return embs.mean(0)

def nv_source_clips():
    import pandas as pd
    inten = pd.read_csv(f"{BASE}/data/neurovoz/extracted_features/"
                        f"intensity_features_neurovoz.csv")
    ki = inten["tot_name"].map(lambda c: c.rsplit("/", 1)[-1][:-4])
    rows = [(stem, f"{BASE}/data/neurovoz/audios/{stem}.wav")
            for stem in ki if "PETACA" in stem]
    return sorted(set(rows))

def pc_target_clips(limit=None, seed=42):
    import pandas as pd
    rng = np.random.RandomState(seed)
    op = {os.path.basename(w): w for w in
          glob.glob(f"{BASE}/data/ParkCeleb/**/*.opus", recursive=True)}
    wv = {os.path.basename(w): w for w in
          glob.glob(f"{BASE}/data/ParkCeleb/**/*.wav", recursive=True)}
    d = pd.read_csv(f"{BASE}/data/parkceleb_extracted/"
                    f"syllable_features_parkceleb.csv")
    af = d["audio_file"].astype(str)
    paths, stems = [], []
    for f in af:
        p = op.get(f.replace(".wav", ".opus")) or wv.get(f)
        if p is not None:
            paths.append(p); stems.append(f[:-4])
    if limit and len(stems) > limit:
        idx = rng.choice(len(stems), size=limit, replace=False)
        paths = [paths[i] for i in idx]; stems = [stems[i] for i in idx]
    return list(zip(stems, paths))

def run(clips, outdir, fe, m):
    os.makedirs(outdir, exist_ok=True)
    done = set(f[:-4] for f in os.listdir(outdir))
    t0 = time.time(); done_n = 0
    for stem, path in clips:
        if stem in done:
            continue
        try:
            x, sr = librosa.load(path, sr=16000)
            if len(x) < 1600:
                continue
            v = embed_clip(fe, m, x, sr)
            np.save(f"{outdir}/{stem}.npy", v)
            done_n += 1
            if done_n % 25 == 0:
                rate = (time.time() - t0) / max(done_n, 1)
                print(f"  {done_n} clips, {rate:.1f}s/clip, "
                      f"ETA {rate*(len(clips)-done_n)/60:.0f} min", flush=True)
        except Exception as e:
            print(f"  FAIL {path}: {type(e).__name__} {e}", flush=True)

if __name__ == "__main__":
    which = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    fe, m = load_model()
    if which == "NV":
        clips = nv_source_clips()
        print(f"NV source clips: {len(clips)}")
        run(clips, f"{BASE}/data/ssl_nv", fe, m)
    else:
        clips = pc_target_clips(limit)
        print(f"PC target clips: {len(clips)}")
        run(clips, f"{BASE}/data/ssl_pc", fe, m)
    print("done")
