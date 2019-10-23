# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:36:01 2019

@author: CarrieLai
"""

import librosa
import numpy as np
import math

sr = 16000
n_fft = math.floor(sr*0.025)
hop_length = math.floor(sr*0.01)
# This is a magic number. The name came from number of samples taken and the
#  hope length when features, e.g. MFCC, spectral_centroid, etc., are being
#  extracted.
timeseries_length = 130
# These are constants for audio feature extractions. e.g. MFCC, etc.
hop_length = 512

def extract_audio_features_list(wave,filename,label):
    # This is a holder to the data that will be extracted from the audio files.
    #  The shape is (num_audio_files x timeseries_length x num_feats) = (1000 x 1293 x 40)
    # This will be features that I will return later.
    data = np.zeros((len(filename), timeseries_length, 40), dtype=np.float64)

    # Iterate over all audio file names.
    for file0 in range(len(filename)):
        # Load the audio file using librosa, to get the frames.
        y = wave[file0]

        # Extract MFCC features. the output will be a (20, timeseries_length).
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=20)

        # Extract spectral_centroid features. the output will be a (20, timeseries_length).
        spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)

        # Extract chroma_stft features. the output will be a (20, timeseries_length).
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

        # Extract spectral_contrast features. the output will be a (20, timeseries_length).
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

        mfcc = mfcc.T
        spectral_center = spectral_center.T
        chroma = chroma.T
        spectral_contrast = spectral_contrast.T

        mfcc = np.pad(mfcc, ((0, timeseries_length-mfcc.shape[0]), (0, 0)), 'edge')
        spectral_center = np.pad(spectral_center, ((0, timeseries_length-spectral_center.shape[0]), (0, 0)), 'edge')
        chroma = np.pad(chroma, ((0, timeseries_length-chroma.shape[0]), (0, 0)), 'edge')
        spectral_contrast = np.pad(spectral_contrast, ((0, timeseries_length-spectral_contrast.shape[0]), (0, 0)), 'edge')

        # Place the received data in the data holder.
        data[file0, :, 0:20] = mfcc[0:timeseries_length, :]
        data[file0, :, 20:21] = spectral_center[0:timeseries_length, :]
        data[file0, :, 21:33] = chroma[0:timeseries_length, :]
        data[file0, :, 33:40] = spectral_contrast[0:timeseries_length, :]

    # return the tuple containing (features, labels) for all the audio files in
    return (data, np.expand_dims(np.asarray(label), axis=1))