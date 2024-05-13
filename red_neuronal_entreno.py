import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

EPOCHS = 1_000

# Cargar los datos
data = pd.read_csv("merged_data.csv")

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
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),  # Convolutional layer: 64 filtros de 3x1 para extraer características
    MaxPooling1D(2),  # Max pooling layer: Reducir la dimensión de las características para obtener las más importantes
    Conv1D(64, 3, activation='relu'),  # Convolutional layer: 64 filtros de 3x1 para extraer características
    MaxPooling1D(2),  # Max pooling layer: Reducir la dimensión de las características para obtener las más importantes
    Flatten(),  # Aplanar las características para poder conectarlas a una capa densa
    Dense(64, activation='relu'),  # Capa densa con 64 neuronas y función de activación ReLU
    Dense(1)  # Capa densa con 1 neurona para la regresión
])

# Función de pérdida
loss_fn = 'mean_squared_error'

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['mean_squared_error']
)

# Entrenar el modelo
model.fit(X_train_scaled, y_train, epochs=EPOCHS, batch_size=32, validation_split=0.2)

# Evaluar el modelo
mse = model.evaluate(X_test_scaled, y_test)
print("Error cuadrático medio en el conjunto de prueba:", mse)

# Predecir un ejemplo
example = X_test_scaled[0].reshape(1, -1)
prediction = model.predict(example)

# comparar la predicción con el valor real
actual_value = y_test.iloc[0]
print("Valor real:", actual_value)
print("Predicción:", prediction[0][0])

# Exportar el modelo
model.save("red_neuronal.h5")
