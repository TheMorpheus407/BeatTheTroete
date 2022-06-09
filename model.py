import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

num_classes = 2

model = Sequential()
model.add(layers.experimental.preprocessing.Resizing(250, 350, interpolation="bilinear"))
#model.add(layers.experimental.preprocessing.Resizing(244, 244, interpolation="bilinear"))
model.add(layers.Conv2D(4, 3, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(8, 3, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(16, 1, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(2, strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])