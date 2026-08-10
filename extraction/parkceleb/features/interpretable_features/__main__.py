from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
from tqdm import tqdm

from features.read_audio_files import read_audio_files
from features.interpretable_features.collectors import get_pause_features, \
    get_syllable_features, get_intensity_features


def run_main(base_directory, max_duration: int | None = None):
    files = read_audio_files(base_directory)
    print("found", len(files), "files in", base_directory)

    with ProcessPoolExecutor(max_workers=6) as executor:
        steps = [
            ["Pause", "pause_features", partial(get_pause_features, max_duration=max_duration)],
            ["Syllables", "syllable_features", partial(get_syllable_features, max_duration=max_duration)],
            ["Intensity", "intensity_features", partial(get_intensity_features, max_duration=max_duration)],
        ]
                                
        with tqdm(steps) as pbar:
            for description, filename, method in pbar:
                pbar.set_description(description)
                features = []

                try:
                    with tqdm(total=len(files), leave=False) as pbar_2:
                        for feat in executor.map(method, files):
                            pbar_2.update(1)
                            features.append(feat)
                except KeyboardInterrupt:
                    continue
                finally:
                    df = pd.DataFrame(features)
                    df.to_csv(f"{filename}_{max_duration}.csv")

if __name__ == "__main__":
    run_main(base_directory="../zenodo/", max_duration=None)
