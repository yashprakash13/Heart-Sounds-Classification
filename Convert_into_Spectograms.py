# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 11:34:39 2018

@author: Costa
"""


import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os

def graph_spectrogram(wav_file, i):
    rate, data = wavfile.read(wav_file)
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=384, NFFT=512)
    ax.axis('off')
    fig.savefig( str(i) +'.png', dpi=300, frameon='false')

i = 0
files = os.listdir(os.getcwd())
for file in files:
    graph_spectrogram(file, i)
    i = i + 1
    

graph_spectrogram('Un.wav', 1)

