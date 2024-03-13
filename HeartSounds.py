import warnings
from matplotlib_inline.config import InlineBackend

warnings.filterwarnings("ignore")
InlineBackend.figure_format = 'retina'

import os
import pandas as pd
import librosa
import librosa.display
import glob
import matplotlib.pyplot as plt
import IPython.display as ipd

# 16 KHz
SAMPLE_RATE = 16000
# seconds
MAX_SOUND_CLIP_DURATION = 12

set_a = pd.read_csv("set_a.csv")
set_a_timing = pd.read_csv("set_a_timing.csv")
set_b = pd.read_csv("set_b.csv")

frames = [set_a, set_b]
train_ab = pd.concat(frames)

#tüm benzersiz etiketleri al
nb_classes = train_ab.label.unique()

print("Number of training examples =", train_ab.shape[0], "  Number of classes =", len(train_ab.label.unique()))
print(nb_classes)

# kategoriye göre veri dağıtımını görselleştirin
category_group = train_ab.groupby(['label', 'dataset']).count()
plot = category_group.unstack().reindex(category_group.unstack().sum(axis=1).sort_values().index)\
          .plot(kind='bar', stacked=True, title="Number of Audio Samples per Category", figsize=(25,10))
plot.set_xlabel("Category")
plot.set_ylabel("Samples Count");

print('Min samples per category =', min(train_ab.label.value_counts()))
print('Max samples per category =', max(train_ab.label.value_counts()))

normal_file = "set_a/normal__201106111136.wav"

ipd.Audio(filename=normal_file)

import wave
wav = wave.open(normal_file)
print("Sampling (frame) rate =", wav.getframerate())
print("Total samples (frames) =", wav.getnframes())
print("Duration =", wav.getnframes() / wav.getframerate())

# Librosa kullanarak yükleme
y, sr = librosa.load(normal_file, sr=None, duration=5)   # default sampling rate is 22 HZ
dur = len(y) / sr
print("Duration:", dur)
print(y.shape, sr)

# librosa plot
plt.figure(figsize=(16, 3))
plt.plot(y)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show()
murmur_file = "set_a/murmur__201108222231.wav"
y2, sr2 = librosa.load(murmur_file, sr=None, duration=5)
dur = len(y2) / sr2
print("Duration:", dur)
print(y2.shape, sr2)

ipd.Audio(filename=murmur_file)

plt.figure(figsize=(16, 3))
plt.plot(y2)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show()

extrahls_file = "set_a/extrahls__201101070953.wav"
y5, sr5 = librosa.load(extrahls_file, sr=None, duration=5)
dur = len(y5) / sr5
print("Duration:", dur)
print(y5.shape, sr5)

ipd.Audio(filename=extrahls_file)
plt.figure(figsize=(16, 3))
plt.plot(y5)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show()

