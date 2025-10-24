# ============================================================
# CLASIFICACIÓN DE ROSTROS - TRANSFER LEARNING CON MOBILENETV2
# ============================================================
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================
# CONFIGURACIÓN
# ==============================
DATA_DIR = r"C:\Jesus\Auto\Proyecto\Original Images"
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123

# ==============================
# CARGA Y PREPROCESADO DE DATOS
# ==============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(" Clases detectadas: {class_names}")

train_ds = train_ds.map(lambda x, y: (x / 255.0, y)).shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

# ==============================
# VISUALIZAR ALGUNAS IMÁGENES
# ==============================
for imgs, labels in train_ds.take(1):
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(imgs[i])
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
    plt.show()

# ==============================
# MODELO CNN - TRANSFER LEARNING
# ==============================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# ENTRENAMIENTO
# ==============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==============================
# GRÁFICO DE PRECISIÓN
# ==============================
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Precisión del Modelo de Clasificación Facial")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# EVALUACIÓN Y PREDICCIONES
# ==============================
test_loss, test_acc = model.evaluate(val_ds)
print(f"\n📊 Precisión final en validación: {test_acc * 100:.2f}%")

for images, labels in val_ds.take(1):
    preds = model.predict(images)
    plt.figure(figsize=(12, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        pred_idx = np.argmax(preds[i])
        conf = np.max(preds[i]) * 100
        pred_class = class_names[pred_idx]
        true_class = class_names[int(labels[i])]
        color = "green" if pred_class == true_class else "red"
        plt.title(f"{pred_class} ({conf:.1f}%)\nReal: {true_class}", color=color)
        plt.axis("off")
    plt.show()
    break

# ==============================
# PREDICCIÓN DE IMAGEN INDIVIDUAL
# ==============================
def predict_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0) / 255.0

    preds = model.predict(img_array)[0]
    name = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    plt.imshow(img)
    plt.title(f"Predicción: {name} ({confidence:.1f}%)")
    plt.axis("off")
    plt.show()

