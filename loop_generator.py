import pandas as pd
import numpy as np
import librosa
import librosa.display
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.image import resize
from sklearn.model_selection import train_test_split
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, Concatenate, GRU, TimeDistributed
import concurrent.futures
import time
from tensorflow.keras.metrics import AUC
import h5py
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import wave
import scipy.io.wavfile as wavf
import os

class LoopGenerator:
    def __init__(self, swing=.55, velocity_var=.1, note_var=2/12, dir_path="loops"): # swing of .5 is neutral
        self.swing = swing
        self.velocity_var = velocity_var
        self.note_var = note_var
        self.dir_path = dir_path

        try:
            self.df = pd.read_csv('audio_metadata - loops.csv', sep=',')
        except:
            self.df = pd.DataFrame()
    

    def safety_guard(self, pattern: list, audio: list, sr: list):
        # count of audio files should match count of sample rates
        if len(audio) != len(sr):
            raise Exception('Different counts of audio and sample rates')
        
        # different sample rates
        if len(np.unique(sr)) > 1:
            raise Exception('Differing sample rates detected')
        sample_rate = sr[0]
        
        # pattern cannot be empty
        if len(pattern) < 8:
            raise Exception(f'Pattern is too small. Length of {len(self.pattern)} is < 8 steps')
        
        return sample_rate


    def generate_loop(self, pattern: list, audio: list, sr: list, bpm, category):
        sample_rate = self.safety_guard(pattern, audio, sr)
        
        num_of_beats = 2 ** np.random.randint(2, 5)
        beat_time = 60 / bpm  # seconds per beat
        loop_duration = beat_time * num_of_beats
        loop_samples = int(np.ceil(loop_duration * sample_rate))
        new_loop = np.zeros(loop_samples, dtype=np.float32)

        for i, hit in enumerate(pattern):
            if hit > 0:
                for j in range(0, hit):
                    velocity = (1-self.velocity_var) * np.random.random_sample() + self.velocity_var
                    sample_idx = np.random.randint(len(audio))
                    sample = audio[sample_idx]

                    start_idx = int((i * (sample_rate * beat_time / 4)) + ((sample_rate / hit) * j))
                    end_idx = start_idx + len(sample)
                    
                    if end_idx > len(new_loop):
                        diff = end_idx - len(new_loop)
                        z = np.zeros(diff)
                        new_loop = np.append(new_loop, z)

                    new_loop[start_idx:end_idx] += sample
        
        new_loop = self.soft_limit(new_loop)
        self.save_loop(new_loop, category)
        #return new_loop


    def soft_limit(self, loop, threshold=.96, max_val=1):
        abs_x = np.abs(loop)
        sign_x = np.sign(loop)
        
        over = abs_x > threshold
        zeroed_vals = abs_x[over] - threshold
        scaled_vals = 1 - np.exp(1)**-zeroed_vals
        scaled_vals = scaled_vals * (max_val - threshold)
        scaled_vals = scaled_vals + threshold

        
        loop[over] = sign_x[over] * scaled_vals
        return loop
    

    def save_loop(self, loop, category):
        print(loop)
        print(self.dir_path)
        print(f'/{self.dir_path}')
        print('')
        print(os.path.join(self.dir_path, category))
        os.makedirs(f'{self.dir_path}', exist_ok=True)
        os.makedirs(os.path.join(self.dir_path, category), exist_ok=True)

            