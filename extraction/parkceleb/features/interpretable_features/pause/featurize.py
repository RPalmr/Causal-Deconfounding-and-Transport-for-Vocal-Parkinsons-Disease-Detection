                                      
import os
import pandas as pd

from features.interpretable_features.pause.prosody import Voice_Prosody
from features.read_audio_files import read_audio_files
from tqdm import tqdm

'''
Featurize Wrapper for grabbing prosody features for audio stored in a folder
'''

def main_pause(audio, output_path, fsize=20, max_duration=None):
    """
    Featurize Wrapper for grabbing prosody features for audio stored in a folder
    """

    df_list = []                            
    vp = Voice_Prosody()
    all_audios = read_audio_files(audio)

    try:
        for fi in tqdm(all_audios, desc='featurizing'):
            if '.m4a' in fi or ".opus" in fi:
                                          
                feat_dict = vp.featurize_audio(fi, int(fsize), max_duration=max_duration)
                                               
                if feat_dict:
                    df_list.append(pd.DataFrame(feat_dict, index=[0]))                                        
    except KeyboardInterrupt:
        pass

    if df_list:                                             
        df = pd.concat(df_list, ignore_index=True)                          
                                                            
        if output_path:
            if max_duration is not None:
                df.to_csv(os.path.join(output_path, f'pause_{max_duration}' + '.csv'))
            else:
                df.to_csv(os.path.join(output_path, 'pause' + '.csv'))

            print(f"Pause CSV saved (max_duration={max_duration})")
            return df
        else:
            print(f"Pause features extracted (max_duration={max_duration})")
            return df

if __name__ == "__main__":
    main_pause("../zenodo/", output_path="features/out/", max_duration=360)
