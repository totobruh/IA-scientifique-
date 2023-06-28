import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Données d'entraînement
pressure = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
temperature = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(pressure, temperature, test_size=0.2, random_state=42)

# Création du modèle de réseau de neurones
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(1,), activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train, y_train, epochs=100, verbose=0)

# Évaluation du modèle sur l'ensemble de test
loss = model.evaluate(X_test, y_test)
print('Loss:', loss)

# Prédiction de la température pour une nouvelle pression
new_pressure = np.array([5.0])
predicted_temperature = model.predict(new_pressure)
print('Predicted Temperature:', predicted_temperature)
