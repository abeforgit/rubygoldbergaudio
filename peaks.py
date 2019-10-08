import pyaudio
import scipy.io.wavfile
import scipy.signal
from scipy.signal import butter, lfilter, freqz
import wave
import numpy as np
import matplotlib.pyplot as plt
import math
import peakutils

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "filtered.wav"

rate, audData = scipy.io.wavfile.read("filtered.wav")
channel1 = audData[:]

time = np.arange(0, float(audData.shape[0]), 1) / RATE
indexes: np.ndarray = peakutils.indexes(channel1, thres=0.8, min_dist=int(RATE/10))
print(indexes)
plt.figure(1)
plt.subplot(211)
markers = indexes.tolist()
plt.plot(time, channel1, '-gD', markevery=markers, linewidth=0.01, alpha=0.7, color='#ff7f00')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
