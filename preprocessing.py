import cv2
import numpy as np
from utils import unpickle, unserialize
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


def prepare_pixels(data):
    """
    Convert data from integers between 0 - 255 to floats between 0 - 1
    """
    data = data.astype("float32")
    norm = data / 255.0
    return norm


def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
	cv2.normalize(hist, hist)
	return hist.flatten()


def prepare_labels(labels):
    """
    One Hot Encoding
    """
    labels = np.array(labels)
    N = labels.shape[0]
    one_hot = np.zeros((N, 10))

    for index, value in enumerate(labels):
        one_hot[index][value] = 1

    return one_hot


def get_train_data(batch_no):
    train_set = unpickle("cifar10/data_batch_" + str(batch_no))
    train_data = train_set[b"data"]
    train_images = unserialize(train_data)
    labels = train_set[b"labels"]
    images = prepare_pixels(train_images)
    labels = prepare_labels(labels)
    return images, labels


def get_test_data():
    test_set = unpickle("cifar10/test_batch")
    test_data = test_set[b"data"]
    test_images = unserialize(test_data)
    labels = test_set[b"labels"]
    images = prepare_pixels(test_images)
    labels = prepare_labels(labels)
    return images, labels


def get_data_from_tensorflow():
    (images_train, labels_train), (images_test, labels_test) = cifar10.load_data()
    
    # normalize the pixel values
    images_train  = images_train.astype('float32')
    images_test   = images_test.astype('float32')
    images_train /= 255
    images_test  /= 255

    # one hot encoding
    labels_train = to_categorical(labels_train, 10)
    labels_test  = to_categorical(labels_test, 10)
    
    return images_train, labels_train, images_test, labels_test
