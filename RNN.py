import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# EMG Sinyal Verileri
emg_data = pd.read_csv('preprocessed_emg_data.csv')

# Verileri Eğitim Ve Test Kümelerine Ayırma
X_train, X_test, y_train, y_test = train_test_split(emg_data.iloc[:, :-1], emg_data.iloc[:, -1], test_size=0.2, random_state=42)

# Verileri RNN ile uyumlu olacak şekilde yeniden şekillendirme
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# RNN modeli oluşturma
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(units=64, activation='tanh', input_shape=X_train.shape[1:]),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Modeli Derleme
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli Eğitme
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Modeli test setinde değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
