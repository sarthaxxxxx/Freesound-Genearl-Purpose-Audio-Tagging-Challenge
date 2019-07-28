import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint
from cfg import config 

def check_point():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path,'rb') as f:
            tmp=pickle.load(f)
            return tmp
    else:
        return None

def get_conv_model():
    model=Sequential()
    model.add(Conv2D(16,(3,3),activation='relu',strides=(1,1),padding='same',input_shape=input_shape))
    model.add(Conv2D(32,(3,3),activation='relu',strides=(1,1),padding='same'))
    model.add(Conv2D(64,(3,3),activation='relu',strides=(1,1),padding='same'))
    model.add(Conv2D(128,(3,3),activation='relu',strides=(1,1),padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def get_recurrent_model():
    model=Sequential()
    model.add(LSTM(128,return_sequences=True,input_shape=input_shape))
    model.add(LSTM(64,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(128,activation='relu')))
    model.add(TimeDistributed(Dense(64,activation='relu')))
    model.add(TimeDistributed(Dense(32,activation='relu')))
    model.add(TimeDistributed(Dense(16,activation='relu')))
    model.add(Flatten())
    model.add(Dense(10,activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


        

df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples=2*int(df['length'].sum()/0.1)
prob_dist=class_dist/class_dist.sum()
choices=np.random.choice(class_dist.index,p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()


config=config(mode='time')

def build_rand_feat():
    tmp=check_point()
    if tmp:
        return tmp.data[0],tmp.data[1]
    x=[]
    y=[]
    fmax,fmin=float('inf'),-float('inf')
    for i in tqdm(range(n_samples)):
        rand_class=np.random.choice(class_dist.index,p=prob_dist)
        file=np.random.choice(df[df.label==rand_class].index)
        rate,s=wavfile.read('clean/'+file)
        label=df.at[file,'label']
        rand_index=np.random.randint(0,s.shape[0]-config.step)
        sample=s[rand_index:rand_index+config.step]
        X_sample=mfcc(sample,rate,numcep=config.mfcc,nfilt=config.nfilt,nfft=config.nfft)
        fmin=min(np.amin(X_sample),fmin)
        fmax=max(np.amax(X_sample),fmax)
        x.append(X_sample)
        y.append(classes.index(label))
    config.max=fmax
    config.min=fmin 
    x,y=np.array(x),np.array(y)
    x=(x-np.mean(x)/np.std(x))
    if config.mode=='conv':
        x=x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    elif config.mode=='time':
        x=x.reshape(x.shape[0],x.shape[1],x.shape[2])
        
    y=to_categorical(y,num_classes=10)
    config.data=(x,y)
    with open(config.p_path,'wb') as f:
        pickle.dump(config,f,protocol=2)
    return x,y
    

if config.mode == 'conv':
    x,y=build_rand_feat()
    y_feat=np.argmax(y,axis=1)
    input_shape=(x.shape[1],x.shape[2],1)
    model=get_conv_model()
    
elif config.mode =='time':
    x,y=build_rand_feat()
    y_feat=np.argmax(y,axis=1)
    input_shape=(x.shape[1],x.shape[2])
    model=get_recurrent_model()



class_weight=compute_class_weight('balanced',np.unique(y_feat),y_feat)  
checkpoint=ModelCheckpoint(config.model_path,monitor='val_acc',verbose=1,mode='max',save_best_only=True,save_weights_only=False,period=1    )
history=model.fit(x,y,epochs=20,batch_size=32,class_weight=class_weight,verbose=1,validation_split=0.1,callbacks=[checkpoint])
model.save(config.model_path)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()