import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa as lb

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1



df=pd.read_csv('instruments.csv')
df.set_index('fname',inplace=True)

for i in df.index:
    sig,sr=lb.load('AudioFiles/'+i)
    df.at[i,'length']=sig.shape[0]/sr #create a new column with the duration of each file in seconds at that particular index
    
classes=list(np.unique(df.label))
class_distribution=df.groupby(['label'])['length'].mean()

fig,ax=plt.subplots()
ax.set_title('Class Distribution',y=1.08)
ax.pie(class_distribution,labels=class_distribution.index,autopct='%1.1f%%',shadow=False,startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)

signals= {}
mfccs= {}
fft= {}
fbank= {}

def calc_fft(y,rate):
    n=len(y)
    freq=np.fft.rfftfreq(n,d=1./rate)
    Y=abs(np.fft.rfft(y)/n)
    return (Y,freq)

def envelope(signal,sr,threshold):
    mask=[]
    y=pd.Series(signal).apply(np.abs)
    y_mean=y.rolling(window=int(sr/10),min_periods=1,center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

for c in classes:
    wav_file=df[df.label==c].iloc[0,0]
    signal,sr=lb.load('AudioFiles/'+wav_file,sr=44100)
    mask=envelope(signal,sr,0.0005)   
    signal=signal[mask]
    signals[c]=signal
    fft[c]=calc_fft(signal,sr)
    bank=logfbank(signal[:sr],sr,nfilt=26,nfft=1103).T
    fbank[c]=bank
    mfccs[c]=mfcc(signal[:sr],sr,numcep=13,nfilt=26,nfft=1103).T
    
    
plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()
    
    
if len(os.listdir('clean'))==0:
    for f in tqdm(df.fname):
        signal,rate=lb.load('AudioFiles/'+f,sr=16000)
        mask=envelope(signal,rate,0.0005)
        wavfile.write(filename='clean/'+f,rate=rate,data=signal[mask])
        
        
    
    


