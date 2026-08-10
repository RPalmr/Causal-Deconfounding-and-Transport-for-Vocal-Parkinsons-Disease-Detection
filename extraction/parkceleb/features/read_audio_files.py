import os
import subprocess

def read_audio_files(input_dir):
    """
    Reads all .wav audio files from a directory tree.

    Args:
        input_dir (str): Root directory to search for audio files.

    Returns:
        list: A list of file paths for the audio files.
    """
    all_audios = []
    for path, subdirs, files in os.walk(input_dir):
        for name in files:
            if name == "recording_concatenated.aac":
                file_path = os.path.join(path, name)
                                              
                if os.path.getsize(file_path) >= 4_096:
                    all_audios.append(file_path)
                else:
                    print(file_path, os.path.getsize(file_path))

    return all_audios





def make_wav(original_file, new_file, max_duration=None):
    if os.path.exists(new_file):
        os.remove(new_file)

    duration = get_duration(original_file)
    if max_duration is not None and duration > max_duration + 2:
        midpoint = duration / 2
        start = midpoint - (max_duration / 2)

        subprocess.run([
            "ffmpeg", "-i", original_file,                                
            "-ss", f"{start:.1f}",                            
            "-t", f"{max_duration:.1f}",                                        
            "-ar", "48000",                     
            "-ac", "1",                        
            "-acodec", "pcm_s16le",                    
            "-f", "wav",                            
            new_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    else:
        subprocess.run([
            "ffmpeg", "-i", original_file,                                
            "-ar", "48000",                     
            "-ac", "1",                        
            "-acodec", "pcm_s16le",                    
            "-f", "wav",                            
            new_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)