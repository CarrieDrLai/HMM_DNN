# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:56:20 2019

@author: CarrieLai
"""

import os 
import librosa
import numpy as np
from hmmlearn import hmm

from data_generator import GetDateset
from extract_feature import *

path0 = os.getcwd();

##########    Step 1 : Generate Voice Sample    ##########

#path_mp3 = path0 + "\\sample_mp3"
#path_wav = path0 + "\\sample_wav"
#GetDateset.DataGenerator(path_mp3, path_wav)

##########     Step 2 :  Dataset Preparing      ##########

path_train = path0 + "\\sample_train"
path_test = path0 + "\\sample_test"
#GetDateset.SeperateData(path_wav,path_train,path_test) 
## 75% : 25% = 81 :27
Dataset = GetDateset()
train_filename, train_wave = Dataset.filename_wave(path_train)
test_filename, test_wave = Dataset.filename_wave(path_test)
train_label = Dataset.GetLabel(train_filename)
test_label = Dataset.GetLabel(test_filename)

###########     Step 3 :  Extract Feature     ############

#   MFCC

train_feat, train_label2 = extract_audio_features_list(train_wave,train_filename,train_label)
# (train_data, train_label2)
test_feat, test_label2 = extract_audio_features_list(test_wave,test_filename,test_label)
# (test_data, test_label2)

##########     Step 4 :  Model - HMM & DNN      ##########


#num = feat.shape[0]
#seq_len = feat.shape[1]
#
#feat = np.concatenate(feat, axis=0)
#
#n_states=len(digit_zh)
#
#n_states=9
#GMMHMM = hmm.GMMHMM(n_components=n_states, n_mix=32, algorithm='viterbi',
#                            covariance_type='diag', n_iter=10, tol=0.01,
#                            verbose=True)
#
#GMMHMM.fit(train_feat)
#
#print GMMHMM.predict_proba(feat[:10, :])
#
#
#from hmm.continuous.GMHMM import GMHMM


#melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
#logmelspec = librosa.power_to_db(melspec)
#
#
#model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
#X1 = np.array([[0,1,0,1],[0,0,0,1],[1,0,1,1]])
#model2.fit(X1)
#model2.fit(X2)
#print (model2.startprob_)
#print model2.transmat_
#print model2.emissionprob_
#print model2.score(X2)
#a = train_feat[0].reshape([np.shape(train_feat)[1]*np.shape(train_feat)[2]])
#b = train_feat[1].reshape([np.shape(train_feat)[1]*np.shape(train_feat)[2]])
#X2 = np.array([a])


models=[]
for k in range(9):
    model = hmm.GMMHMM(n_components=3, n_mix=3,covariance_type='diag', n_iter=1000)
    models.append(model)
# n_components : 隐藏状态数目 , n_iter : 最大迭代次数 
#n_mix is the number of components in a GMM, while n_components refers to the number of Markov chain states (or equivalently the number of mixtures)
    
for i in range(9):
    train_i=np.concatenate(train_feat[9*i:9*(i+1)], axis=0)
    models[i].fit(train_i)

from sklearn.externals import joblib
import pickle
model_path="models\\models.pkl"
#joblib.dump(models, model_path)
output = open("models.pkl", 'wb')
s = pickle.dump(models, output)
output.close()
# # 调用模型
a = joblib.load("models.pkl")


s = a[2].score(test_feat[8])#a[2] -4126.0633569445845 a[6] -1675.7401096405808
m = a[6].predict_proba(test_feat[1], lengths=None)
m=[]
for i in range(27):
    m.append(a[6].predict_proba(test_feat[i], lengths=None))

# model.predict(X)
    
    
#DNN输出向量的维度对应HMM中状态的个数，通常每一维输出对应一个绑定的triphone状态。
#训练时，为了得到每一帧语音在DNN上的目标输出值(标准值)，需要通过事先训练好的GMM-HMM
#识别系统在训练语料上进行强制对齐。即要训练一个DNN-HMM声学模型，首先需要训练一个GMM-HMM
#声学模型，并通过基于Viterbi算法的强制对其方法给每个语音帧打上一个HMM状态标签，
#然后以此状态标签，训练一个基于DNN训练算法的DNN模型。最后用DNN模型替换HMM模型中计算
#观察概率的GMM部分，但保留转移概率和初始概率等其他部分。    
    
    
    
from keras.layers import Dense, Activation, Input
    
def DNN(input_shape=(1,130,40), num_classes=9): 
    inputs = Input(input_shape)
    
    x = Dense(128)(inputs)
    x = Activation('relu')(x)
#    x = Dropout(0.5)(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
#    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    #y = Activation('softmax')(x)
    
    model = Model(inputs, y)
    
    return model