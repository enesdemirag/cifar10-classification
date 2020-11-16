from unpickle import *
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder

def prepare_pixels(data):
    """
    Convert data from integers between 0 - 255 to floats between 0 - 1
    """
    data = data.astype('float32')
    norm = data / 255.0
    return norm

def prepare_labels(labels):
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(labels)
    labels = one_hot_encoder.transform(labels)
    return labels


def get_train_data(batch_no):
    train_set = unpickle('cifar10/data_batch_' + str(batch_no))
    train_data = train_set[b'data']
    train_images = unserialize(train_data)
    labels = train_set[b'labels']

    images = prepare_pixels(train_images)
    labels = prepare_labels(labels)
    return images, labels

def get_test_data():
    test_set = unpickle('cifar10/test_batch')
    test_data = test_set[b'data']
    test_images = unserialize(test_data)
    labels = test_set[b'labels']
    
    images = prepare_pixels(test_images)
    labels = prepare_labels(labels)
    return images, labels

def get_data_from_tensorflow():
    (images_train, labels_train), (images_test, labels_test) = cifar10.load_data()
    return images_train, labels_train, images_test, labels_test