from pause.prosody import Voice_Prosody
from speech_rate.syllables import speech_rate
from speech_rate.intensity_feats_extraction_with_speech_rate import get_intensity

def get_pause_features(input_file, max_duration):
    vp = Voice_Prosody()
    features = vp.featurize_audio(input_file, frame_ms=20, max_duration=max_duration)
    return features

def get_syllable_features(input_file, max_duration):
    return speech_rate(input_file, max_duration=max_duration)

def get_intensity_features(input_file, max_duration):
    return get_intensity(input_file, max_duration)[0]