import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

# Dataset path
dataset_path = os.path.join(os.getcwd(), 'dataset')  # Folder should contain subfolders A, B, C...

# Parameters
img_size = (224, 224)  # MobileNetV2 default input size
batch_size = 16

# Load training and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# ✅ Get class names BEFORE caching or shuffling
class_names = train_ds.class_names
num_classes = len(class_names)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Preprocessing layer for MobileNetV2
preprocess_layer = tf.keras.layers.Lambda(preprocess_input)

# Base model (transfer learning)
base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

# Define model
model = Sequential([
    preprocess_layer,
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callback to stop early if validation loss doesn't improve
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# Save the model
model.save("sign_lang_model.h5")
print("✅ Model trained and saved as sign_lang_model.h5")
