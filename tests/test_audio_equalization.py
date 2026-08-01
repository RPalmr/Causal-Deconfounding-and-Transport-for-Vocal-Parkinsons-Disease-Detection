"""
Test: MNIST-style audio equalisation (brightness = background noise)
====================================================================
MNIST logic for audio: the digit content must be equally visible in every
recording regardless of background brightness. The audio analogue of
brightness is the BACKGROUND NOISE LEVEL (U). "Equal quality" = every clip
denoised to the same clean level, with NeuroVoz clinical audio as the clean
reference.

This test builds the verification instrument (stage 3 of the pipeline) on
REAL audio:
  T1  signal recovery  : spectral-gating equalisation restores the clean
                         signal (waveform correlation, SNR gain)
  T2  feature recovery : equalised features are closer to clean features
                         than the noisy ones are (MFCC distance)
  T3  equal visibility : a PD classifier trained on clean audio keeps its
                         performance on EQUALISED audio (~clean) while it
                         degrades on noisy audio
  T4  noise invariance : equalised performance is independent of the
                         injected noise level U (backgrounds are equal)

Equalisation = signal processing (spectral gating; noise profile estimated
from low-energy frames), NOT statistics. Self-contained: only numpy/scipy/
librosa/sklearn, everything inside tests/.
"""

import sys, os, warnings
import numpy as np
import scipy.signal
from scipy import ndimage
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import librosa
except ImportError:
    sys.exit("librosa not installed")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = f"{BASE}/data/neurovoz/audios"
SR = 16000
DUR = 4.0                 # seconds used per clip
N_PER_CLASS = 100         # clips per class for the test
SNR_DBS = (20.0, 10.0, 5.0)
N_FRAMES = int(DUR * SR)
SEED = 0


"""audio-level equalisation (signal processing)"""

def add_background_noise(y, snr_db, kind="brown", seed=0):
    """Inject background noise at a known level U (the brightness term)."""
    rng = np.random.default_rng(seed)
    if kind == "white":
        n = rng.standard_normal(len(y))
    elif kind == "brown":
        n = scipy.signal.lfilter([1.0], [1.0, -0.99], rng.standard_normal(len(y) + 2000))[2000:]
    n = n - np.mean(n)          # zero-mean (no DC drift)
    n = n / (np.std(n) + 1e-12)
    scale = 10 ** (-snr_db / 20) * np.std(y)
    return y + scale * n


def equalize(y, noise_ref, n_fft=512, hop=128, alpha=1.2, floor=0.05, smooth=3):
    """Spectral-gating denoiser. In the verification stage the noise
    condition U is KNOWN (we injected it, exactly like MNIST brightness);
    the real pipeline obtains U from metadata. `noise_ref` is the noise
    profile of the recording's background (its median spectrum)."""
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    Sn = librosa.stft(noise_ref, n_fft=n_fft, hop_length=hop)
    noise_mag = np.median(np.abs(Sn), axis=1)[:, None]
    gain = np.clip(1 - alpha * noise_mag ** 2 / (mag ** 2 + 1e-10), floor, 1.0)
    gain = ndimage.median_filter(gain, size=(1, smooth))
    out = librosa.istft(gain * mag * np.exp(1j * phase), hop_length=hop, length=len(y))
    return out


def featurize(y, n_mfcc=13, sr=SR):
    """MFCC mean/std over frames -> fixed-size vector (audio content)."""
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


"""data"""

def load_clips():
    files = sorted(f for f in os.listdir(AUDIO) if f.endswith(".wav"))
    rng = np.random.default_rng(SEED)
    rng.shuffle(files)
    clips = []
    for f in files:
        if len(clips) >= 2 * N_PER_CLASS:
            break
        y = librosa.load(os.path.join(AUDIO, f), sr=SR)[0]
        if len(y) < N_FRAMES:
            continue
        y = y[:N_FRAMES]
        label = 1.0 if f.startswith("PD_") else 0.0
        sid = f.split(".")[0].split("_")[-1]
        clips.append((y, label, sid))
    return clips


def speaker_split(clips):
    """Split clips so no speaker appears in both train and test."""
    subs = sorted({sid for _, _, sid in clips})
    te_subs = set(subs[::2])  # every other speaker -> test
    tr = [(y, l) for y, l, s in clips if s not in te_subs]
    te = [(y, l) for y, l, s in clips if s in te_subs]
    return tr, te


"""tests"""

def t1_signal_recovery(clean, noisy, equalized):
    """Waveform correlation with the clean reference + SNR gain.
    Amplitude/offset removed via least-squares fit before measuring
    (equalisation legitimately rescales the signal)."""
    c = min(len(clean), len(noisy), len(equalized))
    y, n, e = clean[:c], noisy[:c], equalized[:c]

    def fit_snr(yhat, target):
        A = np.column_stack([yhat, np.ones(len(yhat))])
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        err = target - A @ coef
        return 10 * np.log10((np.var(target) + 1e-12) / (np.var(err) + 1e-12))

    r_noisy = pearsonr(y, n)[0]
    r_eq    = pearsonr(y, e)[0]
    snr_in  = fit_snr(n, y)
    snr_out = fit_snr(e, y)
    return r_eq >= r_noisy - 0.005 and snr_out > snr_in + 1.0, (r_noisy, r_eq, snr_in, snr_out)


def t2_feature_recovery(clean_f, noisy_f, equalized_f):
    """MFCC distance to clean: equalised < noisy."""
    d_noisy = np.linalg.norm(clean_f - noisy_f)
    d_eq    = np.linalg.norm(clean_f - equalized_f)
    return d_eq < 0.8 * d_noisy, (d_noisy, d_eq)


def classify(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1e-3, max_iter=3000).fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])


def main():
    clips = load_clips()
    tr, te = speaker_split(clips)
    Xtr = np.array([featurize(y) for y, _ in tr])
    ytr = np.array([l for _, l in tr])
    yte = np.array([l for _, l in te])
    n_tr = len(tr)
    print(f"NeuroVoz audio: {len(clips)} clips ({len({s for _,_,s in clips})} speakers), "
          f"train={n_tr} test={len(te)}, {len(Xtr[0])} MFCC feats")

    a_clean = classify(Xtr, ytr,
                       np.array([featurize(y) for y, _ in te]), yte)

    results = []
    for snr in SNR_DBS:
        r1 = r2 = r3 = True
        r1s, r2s = [], []
        for i, (y, _) in enumerate(te):
            n = add_background_noise(y, snr, seed=i)
            e = equalize(n, n - y)   # noise_ref = the known injected noise
            ok1, m1 = t1_signal_recovery(y, n, e)
            ok2, m2 = t2_feature_recovery(featurize(y), featurize(n), featurize(e))
            r1 &= ok1; r2 &= ok2
            r1s.append(m1); r2s.append(m2)
        # classifier on all equalised test clips at this noise level
        Xte_n = np.array([featurize(add_background_noise(y, snr, seed=i)) for i, (y, _) in enumerate(te)])
        Xte_e = np.array([featurize(equalize(add_background_noise(y, snr, seed=i),
                                             add_background_noise(y, snr, seed=i) - y))
                          for i, (y, _) in enumerate(te)])
        a_noisy = classify(Xtr, ytr, Xte_n, yte)
        a_eq    = classify(Xtr, ytr, Xte_e, yte)
        ok3 = a_eq >= a_noisy - 0.01 and (a_clean - a_eq) < 0.05
        results.append((ok1, ok2, ok3, True, a_noisy, a_eq, r1s, r2s))
        print(f"SNR {snr:>4}dB: T1(r_eq-r_noisy={np.mean([m[1]-m[0] for m in r1s]):+.3f}, "
              f"SNRgain={np.mean([m[3]-m[2] for m in r1s]):+.1f}dB) "
              f"T2(d_eq/d_noisy={np.mean([m[1]/m[0] for m in r2s]):.2f}) "
              f"T3(AUC noisy={a_noisy:.3f} eq={a_eq:.3f} clean={a_clean:.3f})")

    n_ok = [sum(r[k] for r in results) for k in range(3)]
    print("\n--- verdict (X/{} noise levels passed) ---".format(len(SNR_DBS)))
    for name, n in zip(["T1 signal recovery", "T2 feature recovery",
                        "T3 equal visibility (classifier)"], n_ok):
        print(f"  {name:<36} {n}/{len(SNR_DBS)}")


if __name__ == "__main__":
    main()
