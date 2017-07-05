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

def extract_cepstrum(filename):
    data, samplerate = sf.read(filename)
    mfcc_feature = mfcc(data,samplerate)
    mfcc_data = np.swapaxes(mfcc_feature, 0 ,1)
    return mfcc_data

hello_world = "./hello-world/helloworld.wav"
dev_location = "./dev-clean"
test_location = "./test-clean"

dev_files = [file for file in glob.glob(dev_location + '/**/*.flac', recursive=True)]
test_files = [file for file in glob.glob(test_location + '/**/*.flac', recursive=True)]

dev_files += test_files

max_count = 10
max_samples = 0
count = 1

mfcc_list = []

for dev_file in dev_files:

    print("\rProcessing file ",count, " of ", len(dev_files), end='\r')
    mfcc_data = extract_cepstrum(dev_file)

    if len(mfcc_data[0]) > max_samples:
        max_samples = len(mfcc_data[0])

    mfcc_list += [mfcc_data]

    count += 1

hello_cepstrum = extract_cepstrum(hello_world)

if len(hello_cepstrum[0]) > max_samples:
    max_samples = len(hello_cepstrum[0])

mfcc_list = [hello_cepstrum] + mfcc_list

y_dimension = len(mfcc_list[0])
dev_data = np.zeros((len(mfcc_list),len(mfcc_list[0]),max_samples))

for i in range(0, len(mfcc_list)):
    for j in range(0, y_dimension):
        padded = np.zeros(max_samples)
        padded[:len(mfcc_list[i][j])] = mfcc_list[i][j]
        dev_data[i][j] = padded

labels = [1]
labels += ([0] * (len(mfcc_list) - 1))

model = Sequential()
model.add(LSTM(256, input_shape=(dev_data.shape[1], dev_data.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

with open('model.json','w') as f:
    f.write(model.to_json())

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(dev_data, labels, epochs=20, batch_size=128, callbacks=callbacks_list)
