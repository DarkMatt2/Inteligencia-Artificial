# ===============================================================
# PROYECTO CRISP-DM — FASES 1 a 6
# Caso de Prueba 3: SVHN (Street View House Numbers) (Nivel: Intermedio)
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

# ===============================================================
# FASE 1: COMPRENSIÓN DEL NEGOCIO
# ===============================================================
"""
Objetivo:
Desarrollar una CNN que clasifique correctamente los dígitos (0–9)
en el conjunto de datos SVHN (Street View House Numbers).

Importancia:
Este problema es un puente entre datasets sintéticos (MNIST) y datos del mundo real.
El reconocimiento de dígitos en imágenes reales es clave para aplicaciones como:
- Lectura automática de números de casas o placas
- Procesamiento OCR basado en cámaras
- Mapas inteligentes y sistemas de navegación

Métrica de éxito:
Obtener una precisión ≥ 85% en el conjunto de prueba.
"""

# ===============================================================
# FASE 2: COMPRENSIÓN DE DATOS
# ===============================================================

print("Cargando dataset SVHN (cropped) desde TensorFlow Datasets...")
(ds_train, ds_test), ds_info = tfds.load(
    'svhn_cropped',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

print("✅ Dataset SVHN cargado correctamente.")
print(ds_info)

# Visualizar algunas imágenes
plt.figure(figsize=(9, 9))
for i, (img, label) in enumerate(ds_train.take(9)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Dígito: {label.numpy()}")
    plt.axis('off')
plt.suptitle("Ejemplos del dataset SVHN (Street View House Numbers)", fontsize=14)
plt.show()

# ===============================================================
# FASE 3: PREPARACIÓN DE DATOS
# ===============================================================

def normalizar_imagen(imagen, etiqueta):
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return imagen, etiqueta

batch_size = 64
ds_train = ds_train.map(normalizar_imagen).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(normalizar_imagen).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("✅ Datos normalizados y preparados para el entrenamiento CNN.")

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

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compilación del modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Entrenando modelo CNN en SVHN...")
history = model.fit(ds_train, epochs=10, validation_data=ds_test)

# ===============================================================
# FASE 5: EVALUACIÓN
# ===============================================================

# Gráficos de entrenamiento
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento (SVHN)')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento (SVHN)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluación final
test_loss, test_acc = model.evaluate(ds_test, verbose=0)
print(f"\n✅ Precisión final del modelo en el conjunto de prueba: {test_acc*100:.2f}%")

# Matriz de confusión
y_true = np.concatenate([y.numpy() for x, y in ds_test], axis=0)
y_pred = np.argmax(model.predict(ds_test), axis=-1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap='Blues', values_format='d')
plt.title("Matriz de Confusión — CNN SVHN")
plt.show()

# ===============================================================
# FASE 6: DESPLIEGUE (con imagen aleatoria)
# ===============================================================

# Guardar modelo
model.save("modelo_cnn_svhn.keras")
print("✅ Modelo guardado como 'modelo_cnn_svhn.keras'")

# Cargar modelo entrenado
modelo_cargado = tf.keras.models.load_model("modelo_cnn_svhn.keras")

# Convertir el conjunto de prueba completo a listas (para elegir una imagen aleatoria)
x_test = []
y_test = []
for imgs, labels in ds_test.unbatch():
    x_test.append(imgs.numpy())
    y_test.append(labels.numpy())
x_test = np.array(x_test)
y_test = np.array(y_test)

# Seleccionar índice aleatorio
indice = random.randint(0, len(x_test) - 1)

# Tomar imagen y etiqueta real
img = np.expand_dims(x_test[indice], axis=0)
real = y_test[indice]

# Predicción
prediccion = np.argmax(modelo_cargado.predict(img))

# Mostrar resultado
print("\n🔍 Predicción del modelo para una imagen aleatoria del test:")
print(f"Etiqueta real: {real}")
print(f"Predicción del modelo: {prediccion}")

plt.imshow(x_test[indice])
plt.title(f"Real: {real} — Predicción: {prediccion}")
plt.axis('off')
plt.show()

print("\n✅ Proceso CRISP-DM completado correctamente para SVHN.")
