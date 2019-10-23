# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:40:49 2019

@author: CarrieLai
"""

### Step 1 : Preparing Digit Voice Sample ###

from aip import AipSpeech
import io
import os
from pydub import AudioSegment
import numpy as np
from shutil import copyfile
import librosa


class GetDateset():
    def __init__(self):
        #####   百度 APPID AK SK     ####
        self.APP_ID = '17573702'
        self.API_KEY = 'Z8MRH3495ncIngyMl8do3aqq'
        self.SECRET_KEY = 'dGxLvElq7eMwgbruMj6tzlQnAnvZkps1'
         
        self.client = AipSpeech(self.APP_ID, self.API_KEY, self.SECRET_KEY)
        
        ######  Sample Generation   #####
        self.digit_zh=['一','二','三','四','五','六','七','八','九']
        self.gender=['female','male']
        self.spd=[5,10];
        self.pit=[0,5,10];
        
        ######  Size of Sample    #######
        self.row = 9;
        self.column = 12;

    def DataGenerator(self,path_mp3,path_wav):
        
        ###   Generate Voice Sample   ###
        for gender0 in range(len(self.gender)):
            for digit_zh0 in range(len(self.digit_zh)):
                for spd0 in range(len(self.spd)):
                    for pit0 in range(len(self.pit)):
                        result  = self.client.synthesis(self.digit_zh[digit_zh0], 'zh', 1, {
                            'spd': self.spd[spd0],'pit':self.pit[pit0],'vol': 10,'per':gender0
                        })
            # vol:音量,per：0女声；1男声；3情感合成度逍遥；4情感合成度丫丫
                        
                        filename = str(1+digit_zh0)+'_'+self.gender[gender0]+'_1_pit'+str(self.pit[pit0])+'_'+'spd'+str(self.spd[spd0])
                        path_mp30 = path_mp3 + "\\" + filename
                        
            # Save MP3 File:识别正确返回语音二进制 错误则返回dict
                        if not isinstance(result, dict):
                            with open(path_mp30+'.mp3', 'wb') as f:
                                f.write(result)
            # Convert MP3 file into wav file:
                        with open(path_mp30+'.mp3', 'rb') as f_wav:
                            data = f_wav.read()
                            aud = io.BytesIO(data)
                            sound = AudioSegment.from_file(aud, format='mp3')
                            sound.export(path_wav + "\\" + filename + '.wav' , format="wav")
                           
                    
    ### Step 2 : Dataset Preparing ###
    
    def SeperateData(self,path_wav,path_train,path_test):
        
        path = os.listdir(path_wav)
        reshape_path = []
        reshape_path = np.reshape(path,[self.row,self.column]) #gender num = 2 ; pit num = 3 ; spd num = 2 ; digit num = 9
    
        for row0 in range(self.row):
            path_shuffle0 = reshape_path[row0][np.random.permutation(self.column)]
            for column0 in range(int(self.column*0.75)):
                new_path = path_train+"\\" + path_shuffle0[column0]
                copyfile(path_wav+"\\"+path_shuffle0[column0], new_path)
                
            for column0 in range(int(self.column*0.75),self.column):
                new_path = path_test+ "\\" + path_shuffle0[column0]
                copyfile(path_wav+"\\"+path_shuffle0[column0], new_path)
    
    ### Step 3(a) : Get Filename & Wave ###
    
    def filename_wave(self,path_t=None):
        filename0 = []
        wave = [] 
        file = os.listdir(path_t)
        for file0 in range(len(file)):
            filename00 = file[file0]
            filename0.append(filename00)
            wave0,sr = librosa.load(path_t+"\\"+filename00,sr=16000)
            wave.append(wave0)
        return filename0,wave
    
    ### Step 3(b) : Get Label ###
    
    def GetLabel(self,filename=None):
        label = []
        for name0 in range(len(filename)):
            name = filename[name0]
            label0 = name.split('_')[0]
            label.append(int(label0))
        return label