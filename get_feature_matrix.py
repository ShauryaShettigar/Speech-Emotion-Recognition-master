# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 05:06:25 2020

@author: Shaurya-PC
"""

import librosa
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import soundfile as sf
import pyloudnorm as pyln

def extract_feature(file):
    id = 1  # Song ID
    feature_set = {'ID':[],
                       'SONG_NAME':[],
                       'rmse':[],
                       'zcr':[],
                       'mfcc':[],
                       'mfcc_delta': [],
                       'loudness':[],
                       'tempo': [],
                       'chroma_stft_mean': [],
                       'chroma_cq_mean':[],
                       'beats':[],
                       'chroma_cens_mean': [],
                       'mel_mean': [],
                       'cent_mean': [],
                       'spec_bw_mean': [],
                       'contrast_mean': [],
                       'rolloff_mean':[],
                       'poly_features': [],
                       'tonnetz': [],
                       'harm_mean': [],
                       'perc_mean' : [],
        
                       }

    songname = file
    
    y, sr = librosa.load(songname, duration=60)
    S = np.abs(librosa.stft(y))
    
            
    # Extracting Features
    rmse = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    poly_features = librosa.feature.poly_features(S=S, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)    
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    
    
    #Calculate Loudness
    data, rate = sf.read(songname)
    meter = pyln.Meter(rate) #
    loudness = meter.integrated_loudness(data)
    
     # Transforming Features
    feature_set['ID'].append(id)
    feature_set['SONG_NAME'].append(songname)
    feature_set['rmse'].append(np.mean(rmse))
    feature_set['mfcc'].append(np.mean(mfcc))
    feature_set['mfcc_delta'].append(np.mean(mfcc_delta))
    feature_set['zcr'].append(np.mean(zcr))
    feature_set['loudness'].append(loudness)
    feature_set['tempo'].append(tempo)
    feature_set['beats'].append(sum(beats))
    feature_set['chroma_stft_mean'].append(np.mean(chroma_stft))
    feature_set['chroma_cq_mean'].append(np.mean(chroma_cq))
    feature_set['chroma_cens_mean'].append(np.mean(chroma_cens))
    feature_set['mel_mean'].append(np.mean(melspectrogram))
    feature_set['cent_mean'].append(np.mean(cent))
    feature_set['spec_bw_mean'].append(np.mean(spec_bw))
    feature_set['contrast_mean'].append(np.mean(contrast))
    feature_set['rolloff_mean'].append(np.mean(rolloff))
    feature_set['poly_features'].append(np.mean(poly_features))
    feature_set['tonnetz'].append(np.mean(tonnetz))
    feature_set['harm_mean'].append(np.mean(harmonic))
    feature_set['perc_mean'].append(np.mean(percussive))
    
    
    print(songname)

    
    return pd.DataFrame(feature_set)
            
