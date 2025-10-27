# ===============================================================
# PROYECTO CRISP-DM — FASES 1 a 6
# Caso de Prueba 6: PlantVillage (Nivel: Aplicado)
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===============================================================
# FASE 1: COMPRENSIÓN DEL NEGOCIO
# ===============================================================
"""
Objetivo:
Desarrollar una CNN con transfer learning para detectar enfermedades en plantas usando el conjunto de datos PlantVillage,
abordando un problema real en agricultura para mejorar la detección temprana y reducir pérdidas en cultivos.

Importancia:
Aplicación socialmente relevante en agricultura sostenible, ayudando a agricultores en regiones en desarrollo.
Introduce técnicas avanzadas como transfer learning y fine-tuning para manejar datos reales con variabilidad.

Métrica de éxito:
Lograr al menos un 85% de precisión en el conjunto de prueba, considerando el desbalanceo de clases y variabilidad de imágenes.
"""

# ===============================================================
# FASE 2: COMPRENSIÓN DE DATOS
# ===============================================================

print("Cargando dataset PlantVillage desde TensorFlow Datasets...")
ds, info = tfds.load('plant_village', split='train', with_info=True, as_supervised=True)

# Información básica del dataset
num_images = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
class_names = info.features['label'].names
print(f"✅ Datos cargados correctamente: {num_images} imágenes totales")
print(f"   38 clases (especies de plantas × enfermedades), imágenes a color de distintos tamaños (original ~256x256)")

# Visualizar algunas imágenes
fig = tfds.show_examples(ds, info, rows=3, cols=3)
plt.suptitle("Ejemplos del dataset PlantVillage", fontsize=14)
plt.show()

# ===============================================================
# FASE 3: PREPARACIÓN DE DATOS
# ===============================================================

# Preprocesamiento: Resize a 224x224 (para MobileNetV2), normalizar
IMG_SIZE = 224
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Dividir en train/val/test (80/10/10 approx) usando 70% del dataset total
ds = ds.shuffle(10000)
ds_size = num_images
subset_size = int(0.70 * ds_size) # Use 70% of the total dataset
ds_subset = ds.take(subset_size)

train_size = int(0.8 * subset_size)
val_size = int(0.1 * subset_size)
test_size = subset_size - train_size - val_size


train_ds = ds_subset.take(train_size).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = ds_subset.skip(train_size).take(val_size).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = ds_subset.skip(train_size + val_size).take(test_size).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)


# Data augmentation para combatir desbalanceo y variabilidad
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

def augment(image, label):
    image = data_augmentation(image)
    return image, label

# Aplicar augmentación solo a train
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

# Batching y prefetch
BATCH_SIZE = 32 # Reduced batch size for memory optimization
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("✅ Datos preprocesados: Resize a 224x224, normalizados, con data augmentation en train para desbalanceo.")

# ===============================================================
# FASE 4: MODELADO
# ===============================================================

# Transfer learning: MobileNetV2 preentrenado en ImageNet
base_model = applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                       include_top=False,
                                       weights='imagenet')

# Congelar base inicialmente
base_model.trainable = False

# Añadir cabeza de clasificación
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compilación
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Entrenando modelo con transfer learning en PlantVillage...")
history = model.fit(train_ds,
                    epochs=2,  # Primera fase: 2 épocas con base congelada
                    validation_data=val_ds)

# Fine-tuning: Descongelar últimas capas y entrenar con LR bajo
base_model.trainable = True
for layer in base_model.layers[:100]:  # Congelar primeras 100 capas
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Fine-tuning del modelo...")
history_fine = model.fit(train_ds,
                         epochs=1,  # Segunda fase: Fine-tuning
                         validation_data=val_ds)

# ===============================================================
# FASE 5: EVALUACIÓN
# ===============================================================

# Combinar histories para gráficos
full_history = {}
for key in history.history:
    full_history[key] = history.history[key] + history_fine.history[key]
for key in history_fine.history:
    if key not in history.history:
        full_history[key] = [None] * len(history.history['accuracy']) + history_fine.history[key]

# Gráficos de entrenamiento
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(full_history['accuracy'], label='Entrenamiento')
plt.plot(full_history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(full_history['loss'], label='Entrenamiento')
plt.plot(full_history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluación final en test
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\n✅ Precisión final del modelo en el conjunto de prueba: {test_acc*100:.2f}%")

# Matriz de confusión (sobre test)
y_pred = []
y_true = []
for images, labels in test_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(15, 15)) # Increase figure size
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d', ax=ax) # Pass ax to disp.plot
plt.title("Matriz de Confusión — CNN PlantVillage")
plt.xticks(rotation=90) # Rotate x-axis labels
plt.yticks(rotation=0) # Ensure y-axis labels are not rotated
plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.show()

# ===============================================================
# FASE 6: DESPLIEGUE
# ===============================================================

# Guardar modelo
model.save("modelo_cnn_plantvillage.keras")
print("✅ Modelo guardado como 'modelo_cnn_plantvillage.keras'")

# Cargar modelo y predecir una imagen de prueba
modelo_cargado = tf.keras.models.load_model("modelo_cnn_plantvillage.keras")

# Tomar una imagen de ejemplo del test_ds
for images, labels in test_ds.take(1):
    imagen = images[0].numpy().reshape(1, IMG_SIZE, IMG_SIZE, 3)
    label_real = labels[0].numpy()
    break

prediccion = np.argmax(modelo_cargado.predict(imagen))

print("\n🔍 Predicción del modelo para una imagen de prueba:")
print(f"Etiqueta real: {class_names[label_real]}")
print(f"Predicción del modelo: {class_names[prediccion]}")

plt.imshow(imagen[0])
plt.title(f"Real: {class_names[label_real]} — Predicción: {class_names[prediccion]}")
plt.axis('off')
plt.show()

print("\n✅ Proceso CRISP-DM completado correctamente para PlantVillage.")
