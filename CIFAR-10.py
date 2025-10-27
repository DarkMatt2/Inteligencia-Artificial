# ===============================================================
# PROYECTO CRISP-DM — FASES 1 a 6
# Caso de Prueba 2: CIFAR-10 (Nivel: Intermedio-Bajo)
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===============================================================
# FASE 1: COMPRENSIÓN DEL NEGOCIO
# ===============================================================
"""
Objetivo:
Desarrollar una CNN capaz de reconocer imágenes a color (32x32) en 10 clases del conjunto de datos CIFAR-10.

Importancia:
El reconocimiento de objetos es esencial en visión por computadora: desde conducción autónoma
hasta clasificación de imágenes en sistemas inteligentes.

Métrica de éxito:
Lograr al menos un 75% de precisión en el conjunto de prueba.
"""

# ===============================================================
# FASE 2: COMPRENSIÓN DE DATOS
# ===============================================================

print("Cargando dataset CIFAR-10 desde TensorFlow...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Información básica del dataset
print(f"✅ Datos cargados correctamente: {x_train.shape} entrenamiento, {x_test.shape} prueba")

# Clases de CIFAR-10
class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
               'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

# Visualizar algunas imágenes
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i])])
    plt.axis('off')
plt.suptitle("Ejemplos del dataset CIFAR-10", fontsize=14)
plt.show()

# ===============================================================
# FASE 3: PREPARACIÓN DE DATOS
# ===============================================================

# Normalización: valores entre 0 y 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Aplanar etiquetas
y_train = y_train.flatten()
y_test = y_test.flatten()

# ===============================================================
# FASE 4: MODELADO
# ===============================================================

# Definición de la arquitectura CNN
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

# Compilación del modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Entrenando modelo CNN en CIFAR-10...")
history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                    validation_data=(x_test, y_test))

# ===============================================================
# FASE 5: EVALUACIÓN
# ===============================================================

# Gráficos de entrenamiento
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluación final
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Precisión final del modelo en el conjunto de prueba: {test_acc*100:.2f}%")

# Matriz de confusión
y_pred = np.argmax(model.predict(x_test), axis=-1)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d', xticks_rotation=45)
plt.title("Matriz de Confusión — CNN CIFAR-10")
plt.show()

# ===============================================================
# FASE 6: DESPLIEGUE
# ===============================================================

# Guardar modelo
model.save("modelo_cnn_cifar10.keras")
print("✅ Modelo guardado como 'modelo_cnn_cifar10.keras'")

# Cargar modelo y predecir una imagen de prueba
modelo_cargado = tf.keras.models.load_model("modelo_cnn_cifar10.keras")

imagen = x_test[0].reshape(1, 32, 32, 3)
prediccion = np.argmax(modelo_cargado.predict(imagen))

print("\n🔍 Predicción del modelo para la primera imagen del test:")
print(f"Etiqueta real: {class_names[y_test[0]]}")
print(f"Predicción del modelo: {class_names[prediccion]}")

plt.imshow(x_test[0])
plt.title(f"Real: {class_names[y_test[0]]} — Predicción: {class_names[prediccion]}")
plt.axis('off')
plt.show()

print("\n✅ Proceso CRISP-DM completado correctamente para CIFAR-10.")
