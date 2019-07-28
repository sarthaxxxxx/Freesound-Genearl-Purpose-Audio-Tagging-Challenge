import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.metrics import accuracy_score
from keras.models import load_model

df=pd.read_csv('instruments.csv')
classes=list(np.unique(df.label))
fn2class=dict(zip(df.fname,df.label))
p_path=os.path.join('pickle','conv.pkl')

with open(p_path,'rb') as f:
    config=pickle.load(f)

model=load_model(config.model_path)
def build_predictions(audio_dir):
    y_true=[]
    y_pred=[]
    fn_prob={}
    print('Extracting features from audio....')
    for f in tqdm(os.listdir(audio_dir)):
        rate,signal=wavfile.read(os.path.join(audio_dir,f))
        label=fn2class[f]
        c=classes.index(label)
        y_prob=[]
        for i in range(0,signal.shape[0]-config.step,config.step):
            sample=signal[i:i+config.step]
            x=mfcc(sample,rate,nfft=config.nfft,nfilt=config.nfilt,numcep=config.mfcc)
            x=(x-np.mean(x))/np.std(x)
            
            if config.mode=='conv':
                x=x.reshape(1,x.shape[0],x.shape[1],1)
            elif config.mode=='time':
                x=np.expand_dims(x,axis=0)
            y_hat=model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
        fn_prob[f]=np.mean(y_prob,axis=0).flatten()
        
    return y_true,y_pred,fn_prob
         
                
y_true,y_pred,fn_prob=build_predictions('clean')
score=accuracy_score(y_true=y_true,y_pred=y_pred)


y_probs=[]
for i,row in df.iterrows():
    y_prob=fn_prob[row.fname]
    y_probs.append(y_prob)
    for c,j in zip(classes,y_prob):
        df.at[i,c]=j
 
y_pred=[classes[np.argmax(y)] for y in y_probs]   
df['y_pred']=y_pred
df.to_csv('Predictions.csv',index=False)   

df=pd.read_csv('Predictions.csv')
df.head()
