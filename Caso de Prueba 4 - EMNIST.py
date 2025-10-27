# ===============================================================
# PROYECTO CRISP-DM — FASES 1 a 6
# Caso de Prueba 4: EMNIST (Nivel: Intermedio-Alto)
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===============================================================
# FASE 1: COMPRENSIÓN DEL NEGOCIO
# ===============================================================
"""
Objetivo:
Desarrollar una CNN capaz de clasificar caracteres manuscritos (letras y dígitos)
del conjunto de datos EMNIST Balanced (47 clases).

Importancia:
EMNIST amplía MNIST con letras mayúsculas y minúsculas, permitiendo construir modelos
para reconocimiento óptico de caracteres (OCR) manuscritos, aplicables en:
- Lectura automática de formularios o documentos escaneados
- Digitalización de texto manuscrito
- Entrenamiento de sistemas OCR educativos o de accesibilidad

Métrica de éxito:
Lograr una precisión ≥ 80% en el conjunto de prueba.
"""

# ===============================================================
# FASE 2: COMPRENSIÓN DE DATOS
# ===============================================================

print("📥 Cargando dataset EMNIST Balanced desde TensorFlow Datasets...")
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

print("✅ Dataset EMNIST Balanced cargado correctamente.")
print(ds_info)

# Visualizar algunas imágenes del dataset
plt.figure(figsize=(8, 8))
for i, (img, label) in enumerate(ds_train.take(9)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(tf.squeeze(img), cmap='gray')
    plt.title(f"Etiqueta: {label.numpy()}")
    plt.axis('off')
plt.suptitle("Ejemplos del dataset EMNIST Balanced", fontsize=14)
plt.show()

# ===============================================================
# FASE 3: PREPARACIÓN DE DATOS
# ===============================================================

# Normalización y expansión de canal
def preparar_datos(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, -1)  # (28, 28, 1)
    return img, label

batch_size = 128
ds_train = ds_train.map(preparar_datos).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preparar_datos).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("✅ Datos normalizados y preparados para entrenamiento CNN.")

# ===============================================================
# FASE 4: MODELADO
# ===============================================================

# Definición de arquitectura CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(47, activation='softmax')  # 47 clases para "Balanced"
])

# Compilación
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Entrenando modelo CNN en EMNIST Balanced...")
history = model.fit(ds_train, epochs=10, validation_data=ds_test)

# ===============================================================
# FASE 5: EVALUACIÓN
# ===============================================================

# Gráficos de entrenamiento
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento (EMNIST Balanced)')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento (EMNIST Balanced)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluación final
test_loss, test_acc = model.evaluate(ds_test, verbose=0)
print(f"\n✅ Precisión final del modelo en el conjunto de prueba: {test_acc*100:.2f}%")

# ===============================================================
# MATRIZ DE CONFUSIÓN — VERSIÓN AMPLIADA
# ===============================================================

print("📊 Generando matriz de confusión...")

y_true = np.concatenate([y.numpy() for x, y in ds_test], axis=0)
y_pred = np.argmax(model.predict(ds_test), axis=-1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(40, 40))  # 🔹 Doble o más tamaño
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d', colorbar=True)

plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8)
plt.title("Matriz de Confusión — CNN EMNIST Balanced (Versión Ampliada)")
plt.tight_layout()
plt.show()

# ===============================================================
# FASE 6: DESPLIEGUE
# ===============================================================

# Guardar modelo
model.save("modelo_cnn_emnist_balanced.keras")
print("✅ Modelo guardado como 'modelo_cnn_emnist_balanced.keras'")

# Cargar modelo y predecir una imagen de prueba
modelo_cargado = tf.keras.models.load_model("modelo_cnn_emnist_balanced.keras")

for imagen, etiqueta in ds_test.take(1):
    img = imagen[0:1]
    real = etiqueta[0].numpy()
    prediccion = np.argmax(modelo_cargado.predict(img))
    print("\n🔍 Predicción del modelo para una imagen del test:")
    print(f"Etiqueta real: {real}")
    print(f"Predicción del modelo: {prediccion}")
    plt.imshow(tf.squeeze(imagen[0]), cmap='gray')
    plt.title(f"Real: {real} — Predicción: {prediccion}")
    plt.axis('off')
    plt.show()
    break

print("\n✅ Proceso CRISP-DM completado correctamente para EMNIST Balanced.")
