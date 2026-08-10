from __future__ import annotations
import subprocess
import sys

import mutagen

                                                        
from .vad_helper import read_wave, frame_generator
import webrtcvad
import os
import sys
            
import shutil
from operator import itemgetter
from itertools import groupby
import numpy as np
from features.read_audio_files import make_wav


class Voice_Prosody:
    def featurize_audio(self, original_audio, frame_ms, max_duration: int | None = None):
        '''
        Central API method to call to perform audio featurization.
        '''
                                                                           
                                                                             
                         

                         
        if original_audio.endswith('.wav'):
            audioFile = original_audio
        else:
            audioFile = os.path.dirname(original_audio) + "-" + os.path.basename(original_audio) + ".wav"
            make_wav(original_audio, audioFile, max_duration)

        vad_dict = self.preproc_audio(audioFile, frame_ms)
        feat_dict = dict()
        feat_names = ['Speech_Time','Total_Time','Pause_Time','Pause_Percentage',
            'Pause_Speech_Ratio','Mean_Pause_Length','Pause_Variability']
        for key, value in vad_dict.items():
            speech_time = self.getSpeechTime(value,frame_ms)
            feat_dict[feat_names[0] + '_VADInt_' + str(key)] = speech_time
            relevant_time = self.getRelevantTime(value,frame_ms)
            feat_dict[feat_names[1] + '_VADInt_' + str(key)] = relevant_time
            pause_time = relevant_time - speech_time
            feat_dict[feat_names[2] + '_VADInt_' + str(key)] = pause_time
            if relevant_time == 0:
                pause_percent = 0                             
            else:
                pause_percent = pause_time / relevant_time
            feat_dict[feat_names[3] + '_VADInt_' + str(key)] = pause_percent
            if speech_time == 0:                             
                pause_sp_ratio = 0
            else:
                pause_sp_ratio = pause_time / speech_time
            feat_dict[feat_names[4] + '_VADInt_' + str(key)] = pause_sp_ratio
            mean_pause = self.meanPauseDuration(value,frame_ms)
            feat_dict[feat_names[5] + '_VADInt_' + str(key)] = mean_pause
            pause_var = self.pauseVariability(value,frame_ms)
            feat_dict[feat_names[6] + '_VADInt_' + str(key)] = pause_var
        feat_dict['tot_name'] = original_audio.split('/')[-1]

        if not original_audio.endswith('.wav'):
            os.remove(audioFile)
        return feat_dict

    def preproc_audio(self,audioFile,frame_ms):
        '''
        Preprocessing Audio File into pcm data and gain segments of data
        and map to voice/nonvoice presence
        '''
        vad_dict = dict()
                                                                        
                                         
                                            
                                                              
                                 
                                    
                                      
                                          

                                     
        levels = [1,2,3]               
        audio, sample_rate = read_wave(audioFile)

        for lv in levels:
            lv_dict = dict()
            vad = webrtcvad.Vad(lv)
            frames = list(frame_generator(frame_ms,audio,sample_rate))                          
            for frame in frames:
                lv_dict[round(frame.timestamp,2)] = str(vad.is_speech(frame.bytes,sample_rate))
            vad_dict[lv] = lv_dict
        return vad_dict

    def getSpeechTime(self,v_dict,frame_ms):
        '''
        Returns Total Speech Time
        '''
        if 'True' not in list(v_dict.values()):
            return 0
        tot_time = list(v_dict.values()).count('True') * frame_ms / 1000
        return tot_time

    def getRelevantTime(self,v_dict,frame_ms):
        '''
        Gets time block from first voicing to last voicing
        '''
        keys = list(v_dict.keys())
        values = list(v_dict.values())
        if 'True' not in values:
            return 0
        f_ind = values.index('True')
        l_ind = len(values) - 1 - values[::-1].index('True')
        tot_time = keys[l_ind] + float(frame_ms)/1000 - keys[f_ind]
        return tot_time

    def calculate_pauses(self,v_dict,frame_ms):
        '''
        Calculates pauses. Returns as an array of pauses
        '''
        pauses = []
        keys = list(v_dict.keys())
        values = list(v_dict.values())
        indices = [i for i, x in enumerate(values) if x == 'False']
        for k, g in groupby(enumerate(indices), lambda ix : ix[0] - ix[1]):
            pause =float(len(list(map(itemgetter(1), g)))) * float(frame_ms) / 1000
            pauses.append(pause)
        return pauses

    def meanPauseDuration(self,v_dict,frame_ms):
        '''
        Calculate Mean Pause Duration:
        - Calculate all the pauses in the sound
        - Average by number of pauses.
        '''
        pauses = self.calculate_pauses(v_dict,frame_ms)
        if len(pauses) == 0 or len(pauses) == 1:                                                            
            return 0
        mean_pause = np.average(pauses)
        return mean_pause

    def pauseVariability(self,v_dict,frame_ms):
        '''
        Calculates the variance of the pauses
        - Calculate pauses in sound clip
        - np.var(array)
        '''
        pauses = self.calculate_pauses(v_dict,frame_ms)
        if len(pauses) == 0 or len(pauses) == 1:                                                            
            return 0
        pause_var = np.var(pauses)
        return pause_var

def main():
    pros = Voice_Prosody()
    path = '/home/lazhang/UW_Projects/MHA_Data/AllAudio'
    print(pros.featurize_audio(os.path.join(path,'NLX-1527883573725426010-1527883619077.wav'),20))

if __name__ == '__main__':
    main()
