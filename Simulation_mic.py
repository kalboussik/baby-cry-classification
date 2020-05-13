import numpy as np
import os
import librosa
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Loaded model from disk")

def preprocess(y,sr):
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T,axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000).T,axis=0)
    chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=40).T,axis=0)
    features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(40,5))
    return(features)


sample_rate = 22050
record_seconds = 5

while True:
    print("Ecoute...")
    rec = sd.rec(int(record_seconds * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Prediction.")
    file = preprocess(rec.T[0],sample_rate)
    file=np.reshape(file,(1,40,5,1))
    print("prediction du classe : ", loaded_model.predict_classes(file))
    print("0: laugh , 1: Cry , 2: Noise , 3: Silence")

