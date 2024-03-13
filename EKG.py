import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal as signal
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

dataset = pd.read_csv("EKG-data.csv")
y = [e for e in dataset.hart]

N = len(y)
Fs = 1000
T = 1.0 / Fs
x = np.linspace(0.0, N * T, N)

yf = np.fft.fft(y)
xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

fig_td = plt.figure()
fig_td.canvas.manager.set_window_title('Time domain signals')
fig_fd = plt.figure()
fig_fd.canvas.manager.set_window_title('Frequency domain signals')
ax1 = fig_td.add_subplot(211)
ax1.set_title('Before filtering')
ax2 = fig_td.add_subplot(212)
ax2.set_title('After filtering')
ax3 = fig_fd.add_subplot(211)
ax3.set_title('Before filtering')
ax4 = fig_fd.add_subplot(212)
ax4.set_title('After filtering')

ax1.plot(x, y, color='r', linewidth=0.7)
ax3.plot(xf, 2.0 / N * np.abs(yf[:N // 2]), color='r', linewidth=0.7, label='raw')
ax3.set_ylim([0, 0.2])

b, a = signal.butter(4, 50 / (Fs / 2), 'low')

tempf = signal.filtfilt(b, a, y)
yff = np.fft.fft(tempf)

nyq_rate = Fs / 2.0
width = 5.0 / nyq_rate
ripple_db = 60.0
O, beta = signal.kaiserord(ripple_db, width)
cutoff_hz = 4.0
taps = signal.firwin(O, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero=False)
y_filt = signal.lfilter(taps, 1.0, tempf)
yff = np.fft.fft(y_filt)

ax4.plot(xf, 2.0 / N * np.abs(yff[:N // 2]), color='g', linewidth=0.7)
ax4.set_ylim([0, 0.2])
ax2.plot(x, y_filt, color='g', linewidth=0.7)

hrw = 0.75
fs = 1000
mov_avg = dataset.hart.rolling(int(hrw * fs)).mean()
avg_hr = np.mean(dataset.hart)
mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
mov_avg = [(0.5 + x) for x in mov_avg]
mov_avg = [x * 1.2 for x in mov_avg]
dataset['filt_rollingmean'] = mov_avg

window = []
peaklist = []
listpos = 0

for datapoint in dataset.hart:
    rollingmean = dataset.filt_rollingmean[listpos]

    if (datapoint < rollingmean) and (len(window) < 1):
        listpos += 1
    elif (datapoint > rollingmean):
        window.append(datapoint)
        listpos += 1
    else:
        maximum = max(window)
        beatposition = listpos - len(window) + window.index(max(window))
        peaklist.append(beatposition)
        window = []
        listpos += 1


ybeat = [dataset.hart[x] for x in peaklist]

fig_hr = plt.figure()
fig_hr.canvas.manager.set_window_title('Peak detector')
ax5 = fig_hr.add_subplot(111)
ax5.set_title("Detected peaks in signal")
ax5.plot(dataset.hart, alpha=0.5, color='blue')
ax5.plot(mov_avg, color='green')
ax5.scatter(peaklist, ybeat, color='red')

RR_list = []
cnt = 0
while cnt < len(peaklist) - 1:
    RR_interval = peaklist[cnt + 1] - peaklist[cnt]
    ms_dist = (RR_interval / fs) * 1000.0
    RR_list.append(ms_dist)
    cnt += 1

bpm = 60000 / np.mean(RR_list)
print("\nAverage Heart Beat: %.01f\n" % bpm)
print("Number of peaks: %d" % len(peaklist))


# Zaman alanı sinyalleri penceresi
root = tk.Tk()
root.title('Time domain signals')

canvas_td = FigureCanvasTkAgg(fig_td, master=root)
canvas_td.draw()
canvas_td.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

root.mainloop()


# Frekans alanı sinyalleri penceresi
root2 = tk.Tk()
root2.title('Frequency domain signals')

canvas_fd = FigureCanvasTkAgg(fig_fd, master=root2)
canvas_fd.draw()
canvas_fd.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

root2.mainloop()
