import pandas as pd
import matplotlib.pyplot as plt

# EMG Sinyal Verilerini Yükleme
emg_data = pd.read_csv('preprocessed_emg_data.csv')

# Veri Kümesinden İlk Sinyali Çıkarma
signal = emg_data.iloc[0, :-1]

# Sinyalleri Çizme
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(signal)
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.set_title('EMG Signal')
plt.show()
