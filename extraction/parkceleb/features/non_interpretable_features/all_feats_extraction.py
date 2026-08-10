                       
                       
"""
Feature extraction for non-interpretable approaches
(x-vector, TRILLsson, wav2vec2/HuBERT)

Input recordings assumed to be under rec_path.
Output feature files (.npy) will be saved under feat_path

"""
import numpy as np
import math
import os
import sys
from numpy import save
import librosa
from speechbrain.lobes.models.huggingface_whisper import HuggingFaceWhisper
import torch
from speechbrain.pretrained import EncoderClassifier
import tensorflow as tf
import tensorflow_hub as hub
from transformers import Wav2Vec2ForSequenceClassification, HubertForSequenceClassification, Wav2Vec2FeatureExtractor

def get_all_sub_segment_inds(x, fs=16e3, dur=10):

    """
        get the range of indices that can be used to run get_sub_segment()
        from the given audio signal
        
        - dur: number of seconds in a segment
    """
    N = x.shape[0]                                    
    N_seg = dur*fs                                                            
    ind_range = math.ceil(N/N_seg)                                          
    return ind_range

def get_sub_segment(x, fs=16e3, dur=10, index=0):

    """
        Get a segment of the input signal x
        
        - dur: number of seconds in a segment
        - index: index of the segment counted from the whole signal
    """
                                         
    N = x.shape[0]                                    
    start_pt = int(index*dur*fs)
    end_pt = int(start_pt + dur*fs)
    if end_pt > N:
        end_pt = N

                 
    seg = x[start_pt:end_pt]
                                              
    if seg.shape[0] < (dur*fs):
        pad_len = int((dur*fs)-seg.shape[0])
        seg = np.pad(seg, ((0,pad_len)), 'constant')

    return seg

                                          
def trillsson_extraction(x,m):
    """
    get trillsson embeddings from one audio
    x: input audio (16khz)
    m = trillsson model
    """
                     
    x = x / (max(abs(x)))
                              
    ind_range = get_all_sub_segment_inds(x, fs=16e3, dur=10)                 
    embeddings = np.zeros(shape=(ind_range, 1024))
    for spec_ind in range(ind_range):
        seg = get_sub_segment(x, fs=16e3, dur=10, index=spec_ind)
        seg = tf.expand_dims(seg, 0)                         
        embedding = m(seg)['embedding']         
        embeddings[spec_ind,:] = embedding.numpy()

                                                
    features_tmp = np.mean(embeddings, axis=0)          
    features_tmp = features_tmp.reshape((1,1024))           

    return features_tmp

def whisper_extraction(x,m):                      

    x = x / (max(abs(x)))
    ind_range = get_all_sub_segment_inds(x, fs=16e3, dur=10)                  

    embeddings = np.zeros(shape=(ind_range, 384))         
    for spec_ind in range(ind_range):
        seg = get_sub_segment(x, fs=16e3, dur=10, index=spec_ind)
        seg = torch.from_numpy(seg)
                         
        seg = seg.unsqueeze(0)                         
        hidden_states = m(seg)
                                                       
        embeddings[spec_ind, :] = hidden_states.squeeze().mean(dim=0).detach().numpy()          

                                                
    features_tmp = np.mean(embeddings, axis=0)
                              
    features_tmp = features_tmp.reshape(1, 384)                  

    return features_tmp

                                               
def xvector_extraction(x, classifier):
                     
    x = x / (max(abs(x))) 
    x = torch.tensor(x[np.newaxis,])                                       

                                         
    embeddings = classifier.encode_batch(x)                                                 
    features_tmp = embeddings.squeeze().numpy()
    features_tmp = np.reshape(features_tmp,(1, features_tmp.size))        

    return features_tmp

def wav2vec2_hubert_extraction(x,feature_extractor,model):
    """
    wav2vec2-base-superb-sid or hubert-base-superb-sid mean pooled hidden states extraction, 
    given x sampled at 16kHz
    """
                     
    x = x / (max(abs(x))) 
    
                                
    ind_range = get_all_sub_segment_inds(x, fs=16e3, dur=10)                 
    embeddings = np.zeros(shape=(ind_range, 13, 768))                              
    for spec_ind in range(ind_range):
        seg = get_sub_segment(x, fs=16e3, dur=10, index=spec_ind)
        inputs = feature_extractor(seg, sampling_rate=16000, padding=True, return_tensors="pt")
        hidden_states = model(**inputs).hidden_states                                       
        for layer_num in range(13):                 
            embeddings[spec_ind,layer_num,:] = hidden_states[layer_num].squeeze().mean(dim=0).detach().numpy()         

                                                
    hidden_states_list = []
    for layer_num in range(13):                 
        hidden_layer_avg = np.mean(embeddings[:,layer_num,:], axis=0)         
        hidden_layer_avg = hidden_layer_avg.reshape((1,768))          
        hidden_states_list.append(hidden_layer_avg)

    return hidden_states_list


def feature_extraction_db_extra(sdir, task_inds, id_inds, out_dir, db_name, trill, nls_labels={}):
    """
    Input 
    - sdri: source directory of both PD and HC data
    - task_inds: list[int], which index or indices of the split filename (split by '_') indicates the task name
    - id_inds: list[int], which index or indices of the split filename (split by '_') indicates the subject id
    - trill: if using trillsson instead of x-vector (modified: 0-xvector,1-trillsson,2-wav2vec,2-hubert)
    - out_dir: output directory to store features of each wav file, include part of file name
    - db_name: str of the db name, used for filename when saving all feats
    - nls_labels: dict { subjectID(str)  ->  category(str) }, required only for nls db
    
    (different way to check labels for db_name = nls)
    czech eg. filename = PD_xx_task_Czech.wav, then task_inds = [2], id_inds = [1]
    
    Output
    - features: dict of { subjectID(str)  ->  dict of {task(str) -> xvector[np 1x512]} }, or 1x1024 
            trillsson instead of xvector
    - cats: dict of { subjectID(str)  ->  category(str) }, 
            category is PD or HC/CN/CTRL
    - tasks: set{str}, list of tasks
    - features_PD: PD features, [#feats x feat dim np arrays]
    - features_HC: HC/CN features, [#feats x feat dim np arrays]
    """
                                                       

                                   
    wav_dicts = [f for f in list(os.scandir(sdir)) if f.name.endswith('.wav') if "concatenated" not in f.name]                                                                                                        
    
                      
    if trill == 1:        
        m = hub.KerasLayer('https://tfhub.dev/google/trillsson1/1')                      

    elif trill == 0:          
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")

    elif trill == 2:           
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")

    elif trill == 3:         
        model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-sid")

    elif trill == 4:
        m = HuggingFaceWhisper('openai/whisper-medium', save_path='')
        m.encoder_only = True

    print('Extracting features from all target train/test data...')


    for wav_dict in wav_dicts:
                                               
        filename = wav_dict.name
        print(filename)

                                         
        x, fs = librosa.load(sdir + "/" + wav_dict.name,sr=16000)             
                                         
        if trill == 0:
            print('extracting with -----> XVECTOR')
            features_tmp = xvector_extraction(x, classifier)
            print(features_tmp.shape)
                                             
            save(out_dir+filename[:-4]+'.npy', features_tmp)                 

        elif trill == 1:
            print('extracting with -----> TRILLSSON')
            features_tmp = trillsson_extraction(x,m)
            print(features_tmp.shape)
                                             
            save(out_dir+filename[:-4]+'.npy', features_tmp)                 

        elif trill == 2:
            print('extracting with -----> WAV2VEC')
            hidden_states_list = wav2vec2_hubert_extraction(x, feature_extractor, model)
            print(hidden_states_list[0].shape)
            for layer_num in range(3, 4):                     
                save(out_dir+filename[:-4]+'.npy', hidden_states_list[layer_num])                 

        elif trill == 3:
            print('extracting with -----> HUBERT')
            hidden_states_list = wav2vec2_hubert_extraction(x, feature_extractor, model)
            print(hidden_states_list[0].shape)
                                             
                                        
            for layer_num in range(6, 7):                     
                save(out_dir+filename[:-4]+'.npy', hidden_states_list[layer_num])                 

        elif trill == 4:
            print('extracting with -----> WHISPER')
            features_tmp = whisper_extraction(x, m)
            print(features_tmp.shape)
                                             
            save(out_dir+filename[:-4]+'.npy', features_tmp)


if __name__ == "__main__":
    rec_path = sys.argv[1]
    feat_path = sys.argv[2]
    feat_type = int(sys.argv[3])
                                         
                                                     

    if feat_type == 0:
        print('Extraction with xvector')
    if feat_type == 1:
        print('Extraction with trillsson')
    if feat_type == 2:
        print('Extraction with wav2vec')
    if feat_type == 3:
        print('Extraction with hubert')
    if feat_type == 4:
        print('Extraction with whisper')

    print(rec_path, feat_path)
    feature_extraction_db_extra(sdir=rec_path, task_inds=[2], id_inds=[1, 2], out_dir=feat_path, db_name='', trill=feat_type)
