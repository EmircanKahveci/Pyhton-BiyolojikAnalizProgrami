import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
import csv

# CSV dosyasından EMG verilerini yükleme
veri = []
with open('EMG-data.csv', 'r') as dosya:
    okuyucu = csv.reader(dosya)
    next(okuyucu)  # Başlık satırını atla
    for satir in okuyucu:
        veri.append(float(satir[0]))  # EMG değerlerinin ilk sütunda olduğunu varsayıyoruz

emg = np.array(veri)
zaman = np.array([i / 1000 for i in range(len(emg))])  # Örnekleme hızının 1000 Hz olduğunu varsayıyoruz

burst1 = emg[:1000] + 0.08
burst2 = emg[1500:2500] + 0.08
quiet = emg[1000:1500] + 0.08
emg = np.concatenate([quiet, burst1, quiet, burst2, quiet])
time = np.array([i / 1000 for i in range(0, len(emg), 1)])  # Örnekleme hızı 1000 Hz olarak varsayılmıştır

# PNG dosyalarını EMG-sonuclar dosyasına kaydetme
klasor = 'EMG-sonuclar'
os.makedirs(klasor, exist_ok=True)


# EMG sinyalini çiz
fig = plt.figure()
plt.plot(time, emg)
plt.xlabel('Time (sec)')
plt.ylabel('EMG (a.u.)')
fig_name = 'fig1.png'
fig.savefig(os.path.join(klasor, fig_name))
fig.set_size_inches(w=11,h=7)


time = []
for i in range(0, len(emg), 1):
    i = i/1000
    time.append(i)
    np.array(time)


# EMG sinyalini işle: ortalamayı kaldır
emg_correctmean = emg - np.mean(emg)


# EMG'nin osffset ve ortalama düzeltilmiş değerlerle karşılaştırmasını çizimi
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.subplot(1, 2, 1).set_title('Mean offset present')
plt.plot(time, emg)
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
plt.ylim(-1.5, 1.5)
plt.xlabel('Time (sec)')
plt.ylabel('EMG (a.u.)')

plt.subplot(1, 2, 2)
plt.subplot(1, 2, 2).set_title('Mean-corrected values')
plt.plot(time, emg_correctmean)
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
plt.ylim(-1.5, 1.5)
plt.xlabel('Time (sec)')
plt.ylabel('EMG (a.u.)')

fig.tight_layout()
fig_name = 'fig2.png'
fig.savefig(os.path.join(klasor, fig_name))
fig.set_size_inches(w=11,h=7)


# EMG için bant geçiren filtre oluştur
high = 20/(1000/2)
low = 450/(1000/2)
b, a = sp.signal.butter(4, [high,low], btype='bandpass')

# process EMG signal: filter EMG
emg_filtered = sp.signal.filtfilt(b, a, emg_correctmean)


# EMG sinyalini işle: EMG'yi filtrele
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.subplot(1, 2, 1).set_title('Unfiltered EMG')
plt.plot(time, emg_correctmean)
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
plt.ylim(-1.5, 1.5)
plt.xlabel('Time (sec)')
plt.ylabel('EMG (a.u.)')

plt.subplot(1, 2, 2)
plt.subplot(1, 2, 2).set_title('Filtered EMG')
plt.plot(time, emg_filtered)
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
plt.ylim(-1.5, 1.5)
plt.xlabel('Time (sec)')
plt.ylabel('EMG (a.u.)')

fig.tight_layout()
fig_name = 'fig3.png'
fig.savefig(os.path.join(klasor, fig_name))
fig.set_size_inches(w=11,h=7)


# EMG sinyalini işle: düzelt
emg_rectified = abs(emg_filtered)

# düzeltilmemiş ve düzeltilmiş EMG'nin arsa karşılaştırması
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.subplot(1, 2, 1).set_title('Unrectified EMG')
plt.plot(time, emg_filtered)
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
plt.ylim(-1.5, 1.5)
plt.xlabel('Time (sec)')
plt.ylabel('EMG (a.u.)')

plt.subplot(1, 2, 2)
plt.subplot(1, 2, 2).set_title('Rectified EMG')
plt.plot(time, emg_rectified)
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
plt.ylim(-1.5, 1.5)
plt.xlabel('Time (sec)')
plt.ylabel('EMG (a.u.)')

fig.tight_layout()
fig_name = 'fig4.png'
fig.savefig(os.path.join(klasor, fig_name))
fig.set_size_inches(w=11,h=7)


def filteremg(time, emg, low_pass=10, sfreq=1000, high_band=20, low_band=450):
    """
    time: Time data
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """

    # kesme frekanslarını örnekleme frekansına normalleştir
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)

    # EMG için bant geçiren filtre oluştur
    b1, a1 = sp.signal.butter(4, [high_band, low_band], btype='bandpass')

    # EMG sinyalini işle: EMG'yi filtrele
    emg_filtered = sp.signal.filtfilt(b1, a1, emg)

    # EMG sinyalini işle: düzelt
    emg_rectified = abs(emg_filtered)

    # düşük geçiş filtresi oluştur ve EMG zarfını almak için düzeltilmiş sinyale uygula
    low_pass = low_pass / (sfreq / 2)
    b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
    emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)

    # plot grafikleri
    fig = plt.figure()
    plt.subplot(1, 4, 1)
    plt.subplot(1, 4, 1).set_title('Unfiltered,' + '\n' + 'unrectified EMG')
    plt.plot(time, emg)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    plt.subplot(1, 4, 2)
    plt.subplot(1, 4, 2).set_title(
        'Filtered,' + '\n' + 'rectified EMG: ' + str(int(high_band * sfreq)) + '-' + str(int(low_band * sfreq)) + 'Hz')
    plt.plot(time, emg_rectified)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(-1.5, 1.5)
    plt.plot([0.9, 1.0], [1.0, 1.0], 'r-', lw=5)
    plt.xlabel('Time (sec)')

    plt.subplot(1, 4, 3)
    plt.subplot(1, 4, 3).set_title(
        'Filtered, rectified ' + '\n' + 'EMG envelope: ' + str(int(low_pass * sfreq)) + ' Hz')
    plt.plot(time, emg_envelope)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.ylim(-1.5, 1.5)
    plt.plot([0.9, 1.0], [1.0, 1.0], 'r-', lw=5)
    plt.xlabel('Time (sec)')

    plt.subplot(1, 4, 4)
    plt.subplot(1, 4, 4).set_title('Focussed region')
    plt.plot(time[int(0.9 * 1000):int(1.0 * 1000)], emg_envelope[int(0.9 * 1000):int(1.0 * 1000)])
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlim(0.9, 1.0)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')

    fig_name = 'fig_' + str(int(low_pass * sfreq)) + '.png'
    fig.set_size_inches(w=11, h=7)
    fig.savefig(os.path.join(klasor, fig_name))


# farklı alçak geçiren filtre kesmelerinin ne yaptığını göster
for i in [3, 10, 40]:
    filteremg(time, emg_correctmean, low_pass=i)

