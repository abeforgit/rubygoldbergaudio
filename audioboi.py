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
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("recording")
frames = []


def getLevel(data):
    pass


for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)
print("finished")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

cutOffFrequency = 10001.0
freqRatio = (cutOffFrequency / RATE)

N = int(math.sqrt(0.196196 + freqRatio ** 2) / freqRatio)

rate, audData = scipy.io.wavfile.read("file.wav")
channel1 = audData[:]

win = np.ones(N)
win *= 1.0 / N
filtered = scipy.signal.lfilter(win, [1], channel1).astype(audData.dtype)

cutOffFrequency = 999.0
freqRatio = (cutOffFrequency / RATE)

N = int(math.sqrt(0.196196 + freqRatio ** 2) / freqRatio)

win = np.ones(N)
win *= 1.0 / N
filtered = filtered - scipy.signal.lfilter(win, [1], filtered).astype(audData.dtype)

indexes = peakutils.indexes(filtered, thres=0.8, min_dist=int(RATE/2))
print(indexes)
# create a time variable in seconds
time = np.arange(0, float(audData.shape[0]), 1) / RATE

plt.figure(1)
plt.subplot(211)
plt.plot(time, filtered, linewidth=0.01, alpha=0.7, color='#ff7f00')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(212)
plt.show()

waveFile = wave.open("filtered.wav", 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(filtered))
waveFile.close()

pips = []
start = 0
for index in indexes:
    diff = index - start
    if diff > 2 * RATE / 10 and diff < 3.25 * RATE / 10:
        pips.append(2)
    if diff > 5 * RATE / 10:
        pips.append(0)
    else:
        pips.append(1)

    start = index

print(pips)


