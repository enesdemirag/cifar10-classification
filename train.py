import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def train_model(model, features, labels, epochs, batch_size=32, shuffle=True):
    history = model.fit(x=features, y=labels, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist

def plot_training(epochs, hist):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    x = hist['accuracy']
    plt.plot(epochs[1:], x[1:], label='accuracy')
    plt.legend()
    plt.show()

def save_model(model, path):
    model.save(path)

def load_model(path):
    model = tf.keras.models.load_model(path)
    return model