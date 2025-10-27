# ===============================================================
# PROYECTO CRISP-DM — FASES 1 a 6
# Caso de Prueba 1: Fashion-MNIST (Nivel: Fácil)
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
Crear un modelo de red neuronal convolucional (CNN) capaz de clasificar
correctamente imágenes de prendas de vestir en 10 categorías del dataset Fashion-MNIST.

Importancia:
El reconocimiento de ropa o accesorios es la base para sistemas de recomendación de moda,
inventarios automatizados y aplicaciones de visión artificial en retail.

Métrica de éxito:
Obtener una precisión ≥ 85% en el conjunto de prueba.
"""

# ===============================================================
# FASE 2: COMPRENSIÓN DE DATOS
# ===============================================================

print("Cargando dataset Fashion-MNIST desde TensorFlow...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Información básica del dataset
print(f"✅ Datos cargados correctamente: {x_train.shape} entrenamiento, {x_test.shape} prueba")

# Clases del dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualizar algunas imágenes
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.suptitle("Ejemplos del dataset Fashion-MNIST", fontsize=14)
plt.show()

# ===============================================================
# FASE 3: PREPARACIÓN DE DATOS
# ===============================================================

# Normalización
x_train, x_test = x_train / 255.0, x_test / 255.0

# Redimensionar para incluir el canal (1, escala de grises)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"Datos listos para CNN: {x_train.shape}")

# ===============================================================
# FASE 4: MODELADO
# ===============================================================

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Entrenando modelo CNN en Fashion-MNIST...")
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
plt.title("Matriz de Confusión — CNN Fashion-MNIST")
plt.show()

# ===============================================================
# FASE 6: DESPLIEGUE
# ===============================================================

# Guardar modelo
model.save("modelo_cnn_fashion_mnist.keras")
print("✅ Modelo guardado como 'modelo_cnn_fashion_mnist.keras'")

# Cargar y predecir una imagen
modelo_cargado = tf.keras.models.load_model("modelo_cnn_fashion_mnist.keras")

imagen = x_test[0].reshape(1, 28, 28, 1)
prediccion = np.argmax(modelo_cargado.predict(imagen))

print("\n🔍 Predicción del modelo para la primera imagen del test:")
print(f"Etiqueta real: {class_names[y_test[0]]}")
print(f"Predicción del modelo: {class_names[prediccion]}")

plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Real: {class_names[y_test[0]]} — Predicción: {class_names[prediccion]}")
plt.axis('off')
plt.show()

print("\n✅ Proceso CRISP-DM completado correctamente para Fashion-MNIST.")
