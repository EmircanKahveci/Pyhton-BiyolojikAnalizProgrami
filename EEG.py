import os
import numpy as np
import mne
import csv

# CSV dosyasından EEG verilerini yükleme
veri = []
with open('EEG-data.csv', 'r') as dosya:
    okuyucu = csv.reader(dosya)
    next(okuyucu)  # Başlık satırını atla
    for satir in okuyucu:
        if len(satir) > 0 and satir[0].isdigit():  # Satır boş değilse ve sayısal değer içeriyorsa
            veri.append(float(satir[0]))

veri = np.array(veri)

# Veri setini kullanarak işlemleri gerçekleştirme
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw = raw.pick_types(meg=False, eeg=True, eog=False, exclude='bads')

print(raw.info)

raw.crop(0, 60)
raw.plot()
raw.plot_psd()

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
raw.resample(600)
raw.notch_filter(np.arange(60, 241, 60))
events = mne.find_events(raw)
events = mne.make_fixed_length_events(raw, start=0, stop=10, duration=1.)
events = mne.make_fixed_length_events(raw, start=0, stop=10, duration=1., overlap=0.5)
epochs = mne.Epochs(raw, events, preload=True).pick_types(eeg=True)
epochs['1'].plot()
