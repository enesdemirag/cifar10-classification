# The archive contains the files data_batch_1, data_batch_2, ... as well as test_batch.
# Each of these files is a Python "pickled" object produced with cPickle.
# Each file has 10000 serialized images with labels.
# b'batch_label', b'labels', b'data', b'filenames'.
# In this script there are some useful functions.

import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def unpickle(file):
    """
    file: path of the pickled file
    return: dictionary object
    """
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def unserialize(data):
    """
    data: (N, 3072) array of serialized pixel data
    return: (N, 32, 32, 3) tensor as
    """
    N      = data.shape[0]
    tensor = data.reshape(N, 3, 32, 32).transpose(0, 2, 3, 1)
    return tensor


def plot_training(model):
    x = model.hist["accuracy"]
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.plot(model.epochs[1:], x[1:], label="accuracy")
    plt.legend()
    plt.show()


def save_model(model, path):
    model.model.save(path)


def load_model(path):
    model = tf.keras.models.load_model(path)
    return model
