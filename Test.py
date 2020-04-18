#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import os
import librosa
import numpy as np
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical


# In[46]:


def preprocess(filename):
    y,sr=librosa.load(filename)
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T,axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000).T,axis=0)
    chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=40).T,axis=0)
    features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(40,5))
    return(features)


# In[31]:


data = []
classes = pd.DataFrame()

dirName = "{}\\test".format(os.getcwd());

listOfFiles = list()
categories = list()


for (dirpath, dirnames, filenames) in os.walk(dirName):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        
for elem in listOfFiles:
    print(elem)
    classes = classes.append({'file':elem,'categorie':elem.split('\\')[-2]},ignore_index=True)
classes['cat_id'] = classes.categorie.astype("category").cat.codes
display(classes)
classes.to_csv (r'classes_test.csv', index = False, header=True)

for filename in classes["file"]:
    data.append(preprocess(filename))
display(data)


# In[45]:


X=np.array(data)
x_valid=np.reshape(X,(X.shape[0],40,5,1))
y=list(classes['cat_id'])
y= to_categorical(y, num_classes=4)

print(y.shape)
print(x_valid.shape)


# In[44]:


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

# evaluate the model
scores = loaded_model.evaluate(x_valid, y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

score = loaded_model.predict_classes(x_valid)
print(score)


# In[ ]:




