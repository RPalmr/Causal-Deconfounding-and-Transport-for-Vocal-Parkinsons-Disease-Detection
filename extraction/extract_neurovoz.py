import sys, os
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data' / 'neurovoz'
AUDIO_DIR = DATA / 'audios'
OUT_DIR = DATA / 'extracted_features'

INTENSITY_COLS = [
    'min_intensity', 'relative_min_intensity_time', 'max_intensity',
    'relative_max_intensity_time', 'mean_intensity', 'stddev_intensity',
    'q1_intensity', 'median_intensity', 'q3_intensity', 'voiced_fraction',
    'min_pitch', 'relative_min_pitch_time', 'max_pitch',
    'relative_max_pitch_time', 'mean_pitch', 'stddev_pitch',
    'q1_pitch', 'q3_pitch', 'mean_absolute_pitch_slope',
    'pitch_slope_without_octave_jumps', 'f1_mean', 'f1_std',
    'f2_mean', 'f2_std', 'f3_mean', 'f3_std', 'f4_mean', 'f4_std',
    'f1_median', 'f2_median', 'f3_median', 'f4_median',
    'formant_dispersion', 'average_formant', 'mff', 'fitch_vtl',
    'delta_f', 'vtl_delta_f'
]

PAUSE_COLS = []
for i in range(1, 4):
    PAUSE_COLS.extend([
        'Speech_Time_VADInt_' + str(i),
        'Total_Time_VADInt_' + str(i),
        'Pause_Time_VADInt_' + str(i),
        'Pause_Percentage_VADInt_' + str(i),
        'Pause_Speech_Ratio_VADInt_' + str(i),
        'Mean_Pause_Length_VADInt_' + str(i),
        'Pause_Variability_VADInt_' + str(i)
    ])

SYLLABLE_COLS = [
    'nsyll', 'npause', 'dur(s)', 'phonationtime(s)',
    'speechrate(nsyll / dur)', 'articulation_rate(nsyll/phonationtime)',
    'average_syllable_dur(speakingtime/nsyll)'
]

META_COLS = {'Unnamed: 0', 'tot_name'}

def _stem(x):
    s = str(x).rsplit('/', 1)[-1]
    while s.endswith('.wav'):
        s = s[:-4]
    return s.replace('audios-', '').replace('audios/', '')

def _get_audio_files():
    files = []
    for root, dirs, fnames in os.walk(AUDIO_DIR):
        for f in fnames:
            if f.endswith('.wav'):
                files.append(os.path.join(root, f))
    return sorted(files)

def _extract_intensity_pitch_formants(sound):
    intensity = sound.to_intensity()
    pitch = sound.to_pitch()
    formants = sound.to_formant_burg()

    dur = sound.get_total_duration()
    t_vals = np.linspace(0, dur, 1000)

    int_vals = []
    for t in t_vals:
        try:
            v = intensity.get_value(t)
            if v != 0:
                int_vals.append(v)
        except:
            pass

    pit_vals = []
    for t in t_vals:
        try:
            v = pitch.get_value_at_time(t)
            if v != 0:
                pit_vals.append(v)
        except:
            pass

    f1_vals = []
    f2_vals = []
    f3_vals = []
    f4_vals = []
    for t in t_vals:
        try:
            f1_vals.append(formants.get_value_at_time(1, t))
            f2_vals.append(formants.get_value_at_time(2, t))
            f3_vals.append(formants.get_value_at_time(3, t))
            f4_vals.append(formants.get_value_at_time(4, t))
        except:
            pass

    if len(int_vals) == 0:
        return {c: np.nan for c in INTENSITY_COLS}

    min_int = np.min(int_vals)
    max_int = np.max(int_vals)
    mean_int = np.mean(int_vals)
    std_int = np.std(int_vals)
    q1_int = np.percentile(int_vals, 25)
    med_int = np.median(int_vals)
    q3_int = np.percentile(int_vals, 75)

    voiced_fraction = len(pit_vals) / len(t_vals) if len(t_vals) > 0 else 0

    if len(pit_vals) > 0:
        min_pit = np.min(pit_vals)
        max_pit = np.max(pit_vals)
        mean_pit = np.mean(pit_vals)
        std_pit = np.std(pit_vals)
        q1_pit = np.percentile(pit_vals, 25)
        q3_pit = np.percentile(pit_vals, 75)

        pit_slopes = np.abs(np.diff(pit_vals))
        mean_abs_pit_slope = np.mean(pit_slopes) if len(pit_slopes) > 0 else 0
        pit_slopes_no_jumps = pit_slopes[pit_slopes < np.percentile(pit_slopes, 95)]
        pit_slope_no_jumps = np.mean(pit_slopes_no_jumps) if len(pit_slopes_no_jumps) > 0 else 0
    else:
        min_pit = max_pit = mean_pit = std_pit = q1_pit = q3_pit = np.nan
        mean_abs_pit_slope = pit_slope_no_jumps = np.nan

    f1_mean = np.mean(f1_vals) if len(f1_vals) > 0 else np.nan
    f1_std = np.std(f1_vals) if len(f1_vals) > 0 else np.nan
    f2_mean = np.mean(f2_vals) if len(f2_vals) > 0 else np.nan
    f2_std = np.std(f2_vals) if len(f2_vals) > 0 else np.nan
    f3_mean = np.mean(f3_vals) if len(f3_vals) > 0 else np.nan
    f3_std = np.std(f3_vals) if len(f3_vals) > 0 else np.nan
    f4_mean = np.mean(f4_vals) if len(f4_vals) > 0 else np.nan
    f4_std = np.std(f4_vals) if len(f4_vals) > 0 else np.nan

    f1_med = np.median(f1_vals) if len(f1_vals) > 0 else np.nan
    f2_med = np.median(f2_vals) if len(f2_vals) > 0 else np.nan
    f3_med = np.median(f3_vals) if len(f3_vals) > 0 else np.nan
    f4_med = np.median(f4_vals) if len(f4_vals) > 0 else np.nan

    if len(f1_vals) > 0 and len(f2_vals) > 0:
        formant_disp = np.mean(np.array(f2_vals) - np.array(f1_vals))
        avg_formant = np.mean(np.array(f1_vals) + np.array(f2_vals) + np.array(f3_vals) + np.array(f4_vals)) / 4
    else:
        formant_disp = avg_formant = np.nan

    mff = f1_mean if not np.isnan(f1_mean) else np.nan
    fitch_vtl = 35000 / (4 * f1_mean) if f1_mean and f1_mean > 0 else np.nan
    delta_f = f2_mean - f1_mean if not np.isnan(f1_mean) and not np.isnan(f2_mean) else np.nan
    vtl_delta_f = 35000 / (2 * delta_f) if delta_f and delta_f > 0 else np.nan

    return {
        'min_intensity': min_int, 'relative_min_intensity_time': 0,
        'max_intensity': max_int, 'relative_max_intensity_time': 0,
        'mean_intensity': mean_int, 'stddev_intensity': std_int,
        'q1_intensity': q1_int, 'median_intensity': med_int,
        'q3_intensity': q3_int, 'voiced_fraction': voiced_fraction,
        'min_pitch': min_pit, 'relative_min_pitch_time': 0,
        'max_pitch': max_pit, 'relative_max_pitch_time': 0,
        'mean_pitch': mean_pit, 'stddev_pitch': std_pit,
        'q1_pitch': q1_pit, 'q3_pitch': q3_pit,
        'mean_absolute_pitch_slope': mean_abs_pit_slope,
        'pitch_slope_without_octave_jumps': pit_slope_no_jumps,
        'f1_mean': f1_mean, 'f1_std': f1_std,
        'f2_mean': f2_mean, 'f2_std': f2_std,
        'f3_mean': f3_mean, 'f3_std': f3_std,
        'f4_mean': f4_mean, 'f4_std': f4_std,
        'f1_median': f1_med, 'f2_median': f2_med,
        'f3_median': f3_med, 'f4_median': f4_med,
        'formant_dispersion': formant_disp, 'average_formant': avg_formant,
        'mff': mff, 'fitch_vtl': fitch_vtl,
        'delta_f': delta_f, 'vtl_delta_f': vtl_delta_f
    }

def _extract_pause_features(sound):
    try:
        textgrid = call(sound, 'To TextGrid (silences)', 100, 0, -25, 0.1, 0.1, 'silent', 'sounding')
        intervals = call(textgrid, 'Get number of intervals', 1)
    except:
        return {c: np.nan for c in PAUSE_COLS}

    results = {}
    for vad_idx in range(1, 4):
        speech_time = 0.0
        pause_time = 0.0
        pause_lengths = []
        total_time = sound.get_total_duration()

        for i in range(1, intervals + 1):
            label = call(textgrid, 'Get label of interval', 1, i)
            start = call(textgrid, 'Get start point', 1, i)
            end = call(textgrid, 'Get end point', 1, i)
            dur = end - start
            if label == 'sounding':
                speech_time += dur
            else:
                pause_time += dur
                pause_lengths.append(dur)

        pause_pct = pause_time / total_time if total_time > 0 else 0
        pause_speech_ratio = pause_time / speech_time if speech_time > 0 else 0
        mean_pause = np.mean(pause_lengths) if pause_lengths else 0
        pause_var = np.std(pause_lengths) if len(pause_lengths) > 1 else 0

        prefix = 'VADInt_' + str(vad_idx)
        results['Speech_Time_' + prefix] = speech_time
        results['Total_Time_' + prefix] = total_time
        results['Pause_Time_' + prefix] = pause_time
        results['Pause_Percentage_' + prefix] = pause_pct
        results['Pause_Speech_Ratio_' + prefix] = pause_speech_ratio
        results['Mean_Pause_Length_' + prefix] = mean_pause
        results['Pause_Variability_' + prefix] = pause_var

    return results

def _extract_syllable_features(sound):
    try:
        syllables = call(sound, 'To Syllables', 100, 400, 0.01, 0.1, 0.01, 0.01)
        nsyll = call(syllables, 'Get number of syllables')
    except:
        return {c: np.nan for c in SYLLABLE_COLS}

    dur = sound.get_total_duration()
    try:
        phonation = call(sound, 'Get phonation time', 0, 0, 0.01, 0.02)
    except:
        phonation = dur

    nsyll = float(nsyll)
    npause = max(0, nsyll - 1)
    speechrate = nsyll / dur if dur > 0 else 0
    artic_rate = nsyll / phonation if phonation > 0 else 0
    avg_syll_dur = dur / nsyll if nsyll > 0 else 0

    return {
        'nsyll': nsyll, 'npause': npause, 'dur(s)': dur,
        'phonationtime(s)': phonation, 'speechrate(nsyll / dur)': speechrate,
        'articulation_rate(nsyll/phonationtime)': artic_rate,
        'average_syllable_dur(speakingtime/nsyll)': avg_syll_dur
    }

def _process_file(wav_path):
    try:
        sound = parselmouth.Sound(wav_path)
    except Exception as e:
        print('failed to load ' + wav_path + ': ' + str(e))
        return None

    stem = _stem(wav_path)

    int_feats = _extract_intensity_pitch_formants(sound)
    pause_feats = _extract_pause_features(sound)
    syll_feats = _extract_syllable_features(sound)

    row = {'tot_name': wav_path}
    row.update(int_feats)
    row.update(pause_feats)
    row.update(syll_feats)

    return row

def main():
    print('neurovoz feature extraction start', flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = _get_audio_files()
    print('found ' + str(len(files)) + ' audio files', flush=True)

    intensity_rows = []
    pause_rows = []
    syllable_rows = []

    for i, f in enumerate(files):
        if i % 50 == 0:
            print('processing ' + str(i) + ' / ' + str(len(files)), flush=True)
        row = _process_file(f)
        if row is None:
            continue

        intensity_rows.append({k: row[k] for k in ['tot_name'] + INTENSITY_COLS})
        pause_rows.append({k: row[k] for k in ['tot_name'] + PAUSE_COLS})
        syllable_rows.append({k: row[k] for k in ['tot_name'] + SYLLABLE_COLS})

    intens_df = pd.DataFrame(intensity_rows)
    pause_df = pd.DataFrame(pause_rows)
    syll_df = pd.DataFrame(syllable_rows)

    intens_path = OUT_DIR / 'intensity_features_neurovoz.csv'
    pause_path = OUT_DIR / 'pause_features_neurovoz.csv'
    syll_path = OUT_DIR / 'syllable_features_neurovoz.csv'

    intens_df.to_csv(intens_path, index=True)
    pause_df.to_csv(pause_path, index=True)
    syll_df.to_csv(syll_path, index=True)

    print('saved intensity ' + str(len(intens_df)) + ' rows to ' + str(intens_path), flush=True)
    print('saved pause ' + str(len(pause_df)) + ' rows to ' + str(pause_path), flush=True)
    print('saved syllable ' + str(len(syll_df)) + ' rows to ' + str(syll_path), flush=True)
    print('done', flush=True)

if __name__ == '__main__':
    main()