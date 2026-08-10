import subprocess
import sys
from features.interpretable_features.speech_rate.feature_extraction_utils import *
import pandas as pd
import os
import parselmouth
import soundfile as sf
from tqdm import tqdm
from features.read_audio_files import make_wav, read_audio_files


def extract_intensity_features(input_path, output_path, max_duration=None):
    """
    Extracts intensity, pitch, and formant features from all .wav files in the input directory
    and saves the features to a CSV file.

    Parameters:
        input_path (str): Root directory path where audio files are located.
        output_path (str): Directory path where the output CSV file will be saved.

    Returns:
        pandas.DataFrame: DataFrame containing extracted features.
    """
    all_audios = sorted(read_audio_files(input_path))

    df_tot = []

    try:
        for file in tqdm(all_audios):
            features, failed = get_intensity(file, max_duration)

            df_tot.append(features)

            if failed:
                print("Failed to compute some formant features.")
    except KeyboardInterrupt:
        pass

                                                        
    new_df = pd.concat(df_tot, ignore_index=True)

                            
    new_index_names = ['{}'.format(i) for i in range(1, len(new_df) + 1)]
    new_df.index = new_index_names

                                                         
    if output_path:
        if max_duration is None:
            out_path = os.path.join(output_path, "intensity.csv")
        else:
            out_path = os.path.join(output_path, f"intensity_{max_duration}.csv")
        new_df.to_csv(out_path, index=False)
        return new_df
    else:
        return new_df


def get_intensity(file, max_duration):
    failed = False
    attributes = {}

    if file.endswith(".wav"):
        filename = file
    else:
        filename = os.path.dirname(file) + "-" + os.path.basename(file) + ".wav"
        make_wav(file, filename, max_duration)
    sound = parselmouth.Sound(filename)
    sound.scale_intensity(70)                                    

                                          
    intensity_attributes = get_intensity_attributes(sound)[0]
    pitch_attributes = get_pitch_attributes(sound)[0]

    attributes.update(intensity_attributes)
    attributes.update(pitch_attributes)

                                     
    try:
        formant_attributes = get_formant_attributes(sound)[0]
        attributes.update(formant_attributes)
    except:
        failed = True

                                               
    attributes["tot_name"] = file
    df = pd.DataFrame()
    for attribute in attributes:
        df.at[0, attribute] = attributes[attribute]
    df.at[0, 'tot_name'] = file                                      
    rearranged_columns = df.columns.tolist()[-1:] + df.columns.tolist()[:-1]                   
    df = df[rearranged_columns]

    if not file.endswith(".wav"):
        os.remove(filename)
    return df.iloc[0].to_dict(), failed


if __name__ == "__main__":
    extract_intensity_features("../zenodo/", "features/out/", max_duration=360)