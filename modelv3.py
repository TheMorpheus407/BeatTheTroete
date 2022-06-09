import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

num_classes = 2

model = Sequential()
model.add(layers.experimental.preprocessing.Resizing(500, 700, interpolation="bilinear"))
#model.add(layers.experimental.preprocessing.Resizing(244, 244, interpolation="bilinear"))
model.add(layers.Conv2D(4, 3, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(8, 3, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(16, 1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(2, strides=2))
model.add(layers.Conv2D(16, 1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(2, strides=2))
model.add(layers.Conv2D(32, 1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(2, strides=2))
model.add(layers.Conv2D(32, 1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(2, strides=2))
model.add(layers.Conv2D(64, 1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(2, strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])