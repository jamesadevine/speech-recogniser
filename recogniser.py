import numpy as np
from keras.models import Sequential, model_from_json
import numpy as np
import soundfile as sf
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense
from keras.utils import np_utils

hello_world = "./hello-world/helloworld.wav"
#hello_world = "./test-clean/test-clean/61/70968/61-70968-0000.flac"

def extract_cepstrum(filename):
    data, samplerate = sf.read(filename)
    mfcc_feature = mfcc(data,samplerate)
    mfcc_data = np.swapaxes(mfcc_feature, 0 ,1)
    return mfcc_data

model_json = ''

with open('model.json') as f:
    model_json = f.read()

model = model_from_json(model_json)
model.load_weights('weights-improvement-04-0.0176.hdf5')
hello_wav = extract_cepstrum(hello_world)


dev_files = [file for file in glob.glob('./dev-clean/**/*.flac', recursive=True)]

count = 1
max_samples = 0

mfcc_list = []

for dev_file in dev_files:

    print("\rProcessing file ",count, " of ", len(dev_files), end='\r')
    mfcc_data = extract_cepstrum(dev_file)

    if len(mfcc_data[0]) > max_samples:
        max_samples = len(mfcc_data[0])

    mfcc_list += [mfcc_data]

    count += 1

max_samples = 3494

dev_data = np.zeros((len(mfcc_list),len(mfcc_list[0]),max_samples))

for i in range(0, len(mfcc_list)):
    for j in range(0, 13):
        padded = np.zeros(max_samples)
        padded[:len(mfcc_list[i][j])] = mfcc_list[i][j]
        dev_data[i][j] = padded

X = np.zeros((1, len(hello_wav), max_samples))

for i in range(0,len(hello_wav)):
    temp = np.zeros(max_samples)
    temp[:len(hello_wav[i])] = hello_wav[i]
    X[0][i] = temp

predictions = model.predict(X)
print(predictions)

predictions = model.predict(dev_data)

false_positive_count = 0

for data in predictions:
    if data[0] > 0.3:
        false_positive_count += 1
        print("FP: ",data[0])   


print(false_positive_count, " false predictions")
