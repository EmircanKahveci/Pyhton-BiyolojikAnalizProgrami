import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import CNN

# Önceden işlenmiş EMG sinyalleri verilerini yükleme
emg_data = pd.read_csv('preprocessed_emg_data.csv')

# Verileri eğitim ve test kümelerine ayırma
X = emg_data.iloc[:, :-1]
y = emg_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim verileri üzerinde bir lojistik regresyon modeli eğitme
model = CNN.model
model.fit(X_train, y_train)

# Veri Kümesini Test Etme
y_pred = model.predict(X_test)

#  accuracy, precision, recall, ve f1 score değerlerinin bulunması
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")
