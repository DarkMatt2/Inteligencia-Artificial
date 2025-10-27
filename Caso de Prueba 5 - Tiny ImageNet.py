# ===============================================================
# PROYECTO CRISP-DM — FASES 1 a 6
# Caso de Prueba 5: Tiny ImageNet (Nivel: Avanzado) — TRANSFER LEARNING MEJORADO
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import shutil
import pandas as pd

# ===============================================================
# FASE 1: COMPRENSIÓN DEL NEGOCIO
# ===============================================================
"""
Objetivo:
Clasificar 200 clases en Tiny ImageNet usando TRANSFER LEARNING con MobileNetV2,
obteniendo un modelo eficiente y robusto para tareas de clasificación a gran escala.

Importancia:
Tiny ImageNet es un benchmark intermedio entre datasets pequeños (CIFAR) y ImageNet completo.
Trabajar con Tiny ImageNet permite:
- Evaluar técnicas de Transfer Learning y Fine-tuning en un escenario con muchas clases.
- Desarrollar modelos útiles para sistemas de búsqueda y recuperación de imágenes.
- Mejorar pipelines de visión para aplicaciones de robótica y vehículos autónomos
  que requieren clasificación rápida y de bajo costo computacional.
- Facilitar investigación en data augmentation, calibración de confianza y reducción de dimensión.
- Probar estrategias de despliegue eficientes (modelos compactos, pruning, quantization).

Métrica de éxito:
Alcanzar > 60% de accuracy en validación tras fine-tuning y optimizaciones.
"""

# ===============================================================
# FASE 2: COMPRENSIÓN DE DATOS
# ===============================================================

# Descargar dataset (solo si no existe)
if not os.path.exists('tiny-imagenet-200'):
    !wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip
    !unzip -q tiny-imagenet-200.zip
    print("Dataset descargado y extraído.")

base_dir = 'tiny-imagenet-200'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
val_annotations = os.path.join(val_dir, 'val_annotations.txt')

# Organizar validación por clases
val_images_dir = os.path.join(val_dir, 'images')
organized_val_dir = os.path.join(val_dir, 'organized')

if not os.path.exists(organized_val_dir):
    os.makedirs(organized_val_dir)
    df = pd.read_csv(val_annotations, sep='\t', header=None,
                     names=['file', 'class', 'x1', 'y1', 'x2', 'y2'])
    for _, row in df.iterrows():
        class_dir = os.path.join(organized_val_dir, row['class'])
        os.makedirs(class_dir, exist_ok=True)
        src = os.path.join(val_images_dir, row['file'])
        dst = os.path.join(class_dir, row['file'])
        if os.path.exists(src):
            shutil.move(src, dst)
    print("Validación organizada por clases.")

# Data Augmentation MEJORADO
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# Generadores con mejor resolución
IMG_SIZE = (96, 96)
BATCH_SIZE = 64

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    organized_val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Clases detectadas: {len(train_generator.class_indices)}")

# ===============================================================
# FASE 4: MODELADO — TRANSFER LEARNING MEJORADO
# ===============================================================

base_model = applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(200, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================================================
# CALLBACKS PARA MEJOR ENTRENAMIENTO
# ===============================================================

callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)
]

# ===============================================================
# ENTRENAMIENTO Parte 1
# ===============================================================
print("=" * 60)
print("Parte 1 del Entrenamiento con capas base congeladas")
print("=" * 60)

history1 = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)

# ===============================================================
# ENTRENAMIENTO parte 2: FINE-TUNING
# ===============================================================
print("\n" + "=" * 60)
print("Parte 2: Fine-tuning - Descongelando últimas capas")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)

# ===============================================================
# FASE 5: EVALUACIÓN
# ===============================================================
history_combined = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history_combined['accuracy'], label='Train')
plt.plot(history_combined['val_accuracy'], label='Val')
plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning start')
plt.title('Accuracy (2 Fases de Entrenamiento)')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history_combined['loss'], label='Train')
plt.plot(history_combined['val_loss'], label='Val')
plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--', label='Fine-tuning start')
plt.title('Loss (2 Fases de Entrenamiento)')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"\nPrecisión en validación: {val_acc*100:.2f}%")
print(f"Loss en validación: {val_loss:.4f}")

# ===============================================================
# MATRIZ DE CONFUSIÓN (TOP 10 CLASES)
# ===============================================================
print("\nGenerando matriz de confusión (Top 10 clases)...")

# Predicciones
val_generator.reset()
preds = model.predict(val_generator, verbose=1)
y_true = val_generator.classes
y_pred = np.argmax(preds, axis=1)

# Calcular matriz
cm = confusion_matrix(y_true, y_pred)
class_labels = list(val_generator.class_indices.keys())

# Seleccionar top 10 clases más frecuentes
class_counts = np.bincount(y_true)
top10_idx = np.argsort(class_counts)[-10:]
top10_labels = [class_labels[i] for i in top10_idx]
cm_top10 = cm[np.ix_(top10_idx, top10_idx)]

# Mostrar
disp = ConfusionMatrixDisplay(confusion_matrix=cm_top10, display_labels=top10_labels)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Matriz de Confusión (Top 10 clases más frecuentes)")
plt.show()

# ===============================================================
# FASE 6: DESPLIEGUE
# ===============================================================
model.save("modelo_tinyimagenet_mobilenetv2_optimized.keras")
print("\nModelo guardado como: modelo_tinyimagenet_mobilenetv2_optimized.keras")

# Ejemplo de predicción
val_generator.reset()
x_batch, y_batch = next(val_generator)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    img = x_batch[i]
    pred = model.predict(img.reshape(1, 96, 96, 3), verbose=0)
    pred_class = np.argmax(pred)
    true_class = np.argmax(y_batch[i])
    confidence = np.max(pred) * 100
    img_display = (img - img.min()) / (img.max() - img.min())

    axes[i].imshow(img_display)
    color = 'green' if pred_class == true_class else 'red'
    true_label = list(train_generator.class_indices.keys())[true_class]
    pred_label = list(train_generator.class_indices.keys())[pred_class]
    axes[i].set_title(f"Real: {true_label[:15]}\nPred: {pred_label[:15]}\nConf: {confidence:.1f}%",
                      color=color, fontsize=10, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("PROCESO CRISP-DM COMPLETADO")
print("=" * 60)
print(f"✓ Precisión final: {val_acc*100:.2f}%")
print("✓ Matriz de confusión resumida (Top 10 clases)")
print("✓ Transfer Learning + Fine-tuning optimizado")
