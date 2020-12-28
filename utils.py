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
    x1 = model.hist["accuracy"]
    x2 = model.hist["loss"]

    _ , ax = plt.subplots(1, 2)
    ax[0].set_xlabel("Epoch")
    ax[1].set_xlabel("Epoch")
    ax[0].set_ylabel("Value")
    ax[1].set_ylabel("Value")
    ax[0].set_title("Accuracy")
    ax[1].set_title("Loss")
    ax[0].plot(model.epochs[1:], x1[1:], label="accuracy")
    ax[1].plot(model.epochs[1:], x2[1:], label="loss")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def load_model(path):
    model = tf.keras.models.load_model(path)
    return model
