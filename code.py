#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout


# # Lecture de données

# In[49]:


classes = pd.DataFrame()
listOfFiles = list()
categories = list()

dirName = "{}\\data".format(os.getcwd());

for (dirpath, dirnames, filenames) in os.walk(dirName):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        
for elem in listOfFiles:
    print(elem)
    classes = classes.append({'file':elem,'categorie':elem.split('\\')[-2]},ignore_index=True)
classes['cat_id'] = classes.categorie.astype("category").cat.codes
display(classes)
classes.to_csv (r'classes.csv', index = False, header=True)


# # Traitement des données (preprocessing)

# In[30]:


data = []

for filename in classes["file"]:
    y,sr=librosa.load(filename)
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T,axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000).T,axis=0)
    chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=40).T,axis=0)
    features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(40,5))
    data.append(features)
display(data)


# # Reshaping

# In[41]:


X=np.array(data)
X_2d=np.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
np.savetxt("train_data.csv", X_2d, delimiter=",")
y=list(classes['cat_id'])
x_train, x_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.25, random_state=42)
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[43]:


y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)
y_train.shape,y_test.shape


# In[44]:


x_train=np.reshape(x_train,(x_train.shape[0], 40,5,1))
x_test=np.reshape(x_test,(x_test.shape[0], 40,5,1))
x_train.shape,x_test.shape


# # CNN Model

# In[46]:


model=Sequential()
model.add(Conv2D(64,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(40,5,1)))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(128,kernel_size=5,strides=1,padding="same",activation="relu"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(4,activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,batch_size=50,epochs=30,validation_data=(x_test,y_test))


# # Sauvegarde du modele

# In[47]:


model.save_weights("model.h5")
print("Saved model to disk")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# In[38]:





# In[ ]:




