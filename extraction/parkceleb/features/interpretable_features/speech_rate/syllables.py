                     
                       

                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
                                                                           
 
                                                                    
                      
                                                                                 
                                                                              
                                               
                                                                              
                                         
                                                                                    
               
                                                        
                                                                  
                                        
                                                      
                                                                            
                 
                                         
                                                                          
                                    
                                                         
                                                              
                                                    
                                                                                      
                                                                               
 
                                                           
                                                                           


                                                         
                                                  
                                                  
                                               
                                                                         
                                                                                 
 

                                                
                                                             

import sys
import math
import pandas as pd
import parselmouth
import os
from features.read_audio_files import read_audio_files, make_wav
from parselmouth.praat import call
from tqdm import tqdm


def speech_rate(original_filename, max_duration=None):

    """
       Calculate speech rate, articulation rate, and average syllable duration from an audio file.

       Parameters:
           filename (str): Path to the audio file.

       Returns:
           dict: A dictionary containing the calculated speech metrics.
       """
    if original_filename.endswith(".wav"):
        filename = original_filename
    else:
        filename = os.path.dirname(original_filename) + "-" + os.path.basename(original_filename) + ".wav"
        make_wav(original_filename, filename, max_duration)

    silencedb = -25
    mindip = 2
    minpause = 0.3
    sound = parselmouth.Sound(filename)
    originaldur = sound.get_total_duration()
    intensity = sound.to_intensity(50)
    start = call(intensity, "Get time from frame number", 1)
    nframes = call(intensity, "Get number of frames")
    end = call(intensity, "Get time from frame number", nframes)
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

                                                                                    
    max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

                                  
    threshold = max_99_intensity + silencedb
    threshold2 = max_intensity - max_99_intensity
    threshold3 = silencedb - threshold2
    if threshold < min_intensity:
        threshold = min_intensity

                                            
    textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
    silencetier = call(textgrid, "Extract tier", 1)
    silencetable = call(silencetier, "Down to TableOfReal", "sounding")
    npauses = call(silencetable, "Get number of rows")
    speakingtot = 0
    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = call(silencetable, "Get value", pause, 1)
        endsound = call(silencetable, "Get value", pause, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur

    intensity_matrix = call(intensity, "Down to Matrix")
                                            
    sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
                                                                                           
                                               
    intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
    intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
    point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
                                         
    numpeaks = call(point_process, "Get number of points")
    t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

                                      
    timepeaks = []
    peakcount = 0
    intensities = []
    for i in range(numpeaks):
        value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
        if value > threshold:
            peakcount += 1
            intensities.append(value)
            timepeaks.append(t[i])

                                                                     
                                             
    validpeakcount = 0
    currenttime = timepeaks[0]
    currentint = intensities[0]
    validtime = []

    for p in range(peakcount - 1):
        following = p + 1
        followingtime = timepeaks[p + 1]
        dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
        diffint = abs(currentint - dip)
        if diffint > mindip:
            validpeakcount += 1
            validtime.append(timepeaks[p])
        currenttime = timepeaks[following]
        currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

                                
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedcount = 0
    voicedpeak = []

    for time in range(validpeakcount):
        querytime = validtime[time]
        whichinterval = call(textgrid, "Get interval at time", 1, querytime)
        whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(querytime)
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

                                                                            
                      
    timecorrection = originaldur / intensity_duration

                                     
    call(textgrid, "Insert point tier", 1, "syllables")
    for i in range(len(voicedpeak)):
        position = (voicedpeak[i] * timecorrection)
        call(textgrid, "Insert point", 1, position, "")

                    
    speakingrate = voicedcount / originaldur
    articulationrate = voicedcount / speakingtot
    npause = npauses - 1
                                    
    if voicedcount != 0:
        asd = speakingtot / voicedcount
    else:
                                                      
                                                                             
        print('handling 0 voiced count --> check file')
        asd = 0
    speechrate_dictionary = {'tot_name': filename,
                             'nsyll': voicedcount,
                             'npause': npause,
                             'dur(s)': originaldur,
                             'phonationtime(s)': intensity_duration,
                             'speechrate(nsyll / dur)': speakingrate,
                             "articulation_rate(nsyll/phonationtime)": articulationrate,
                             "average_syllable_dur(speakingtime/nsyll)": asd}

    if not original_filename.endswith(".wav"):
        os.remove(filename)
    return speechrate_dictionary


def main_syllables(input_dir, output_dir, max_duration=None):
    cols = ['tot_name', 'nsyll', 'npause', 'dur(s)', 'phonationtime(s)', 'speechrate(nsyll / dur)',
            'articulation_rate(nsyll/phonationtime)', 'average_syllable_dur(speakingtime/nsyll)']

    datalist = []
    all_audios = read_audio_files(input_dir)
    try:
        for file in tqdm(all_audios, desc='extract syllables'):
                        
            speechrate_dictionary = speech_rate(file, max_duration=max_duration)                                                      
            datalist.append(speechrate_dictionary)
    except KeyboardInterrupt:
        ...

    df = pd.DataFrame(datalist, columns=cols)
    if output_dir:
        if max_duration is None:
            out_path = os.path.join(output_dir, 'speechrate_data.csv')
        else:
            out_path = os.path.join(output_dir, f'speechrate_data_{max_duration:.0f}.csv')
        df.to_csv(out_path, index=False)
        print("Syllable-based CSV saved.")
        return df
    else:
        print("Syllable-based extracted.")
        return df

if __name__ == "__main__":
    main_syllables(input_dir="../zenodo/", output_dir="features/out/", max_duration=360)
