import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def create_mlp_model(learning_rate):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Dense(units=3072, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))     
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  loss=tf.keras.losses.MeanSquaredError,
                  metrics=['accuracy'])

    return model