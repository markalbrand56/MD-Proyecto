import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

EPOCHS = 20_0

# Cargar los datos
data = pd.read_csv("merged_data.csv")
print(f"Filas: {data.shape[0]}, Columnas: {data.shape[1]}")

# excluir las filas donde 'Year' es 2017 o mayor
data = data[data["Year"] < 2017]
print(f"Filas: {data.shape[0]}, Columnas: {data.shape[1]}")

# Dividir los datos en características (X) y la variable objetivo (y)
X = data.drop(columns=["Depressive disorder rates (number suffering per 100,000)", "Entity", "Code"])
y = data["Depressive disorder rates (number suffering per 100,000)"]


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construir el modelo de red neuronal
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Función de pérdida
loss_fn = 'mean_squared_error'
early_stopping = tf.keras.callbacks.EarlyStopping(patience=500, restore_best_weights=True, monitor='val_mean_squared_error')

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['mean_absolute_error', 'mean_squared_error']
)

# Entrenar el modelo
model.fit(X_train_scaled, y_train, epochs=EPOCHS, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluar el modelo
results = model.evaluate(X_test_scaled, y_test)

mae = results[1]
mse = results[2]

print(f"\nError absoluto medio en el conjunto de prueba: {mae}")
print(f"Error cuadrático medio en el conjunto de prueba: {mse}\n")

# Predecir un ejemplo
example = X_test_scaled[0].reshape(1, -1)
prediction = model.predict(example)

# comparar la predicción con el valor real
actual_value = y_test.iloc[0]
print("Valor real:", actual_value)
print("Predicción:", prediction[0][0])

# Exportar el modelo
model.save("output/red_neuronal_1.h5")
