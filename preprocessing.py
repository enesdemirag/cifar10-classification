from unpickle import *


def prepare_pixels(data, labels):
    """
    Convert data from integers between 0 - 255 to floats between 0 - 1
    """
    labels = np.array(labels, dtype='float32')
    data = data.astype('float32')
    norm = data / 255.0
    return norm, labels

def get_train_data(batch_no):
    train_set = unpickle('cifar10/data_batch_' + str(batch_no))
    train_data = train_set[b'data']
    train_images = unserialize(train_data)
    labels = train_set[b'labels']

    images, labels = prepare_pixels(train_images, labels)
    return images, labels

def get_test_data():
    test_set = unpickle('cifar10/test_batch')
    test_data = test_set[b'data']
    test_images = unserialize(test_data)
    labels = test_set[b'labels']
    
    images, labels = prepare_pixels(test_images, labels)
    return images, labels
