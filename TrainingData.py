import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Önceden işlenmiş EMG sinyalleri verilerini yükleme
emg_data = pd.read_csv('preprocessed_emg_data.csv')

# Verileri eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(emg_data.iloc[:, :-1], emg_data.iloc[:, -1], test_size=0.2, random_state=42)

# Verileri Rastgele Eğitme
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Modeli test setinde değerlendirme
test_acc = clf.score(X_test, y_test)
print('\nTest accuracy:', test_acc)