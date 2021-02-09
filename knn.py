# In this script Kth Nearest Neighbor (Knn) machine learning algorithm used on dataset.csv
# This dataset consist of 1000 samples with 26 features each
# https://scikit-learn.org/stable/modules/neighbors.html

import random
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from preprocessing import get_data_from_tensorflow, image_to_feature_vector, extract_color_histogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# Preprocessing
images_train, labels_train, images_test, labels_test = get_data_from_tensorflow()

labels_train = np.argmax(labels_train, axis=1)
labels_test = np.argmax(labels_test, axis=1)

# Loop over the training images
raw_images = []
features = []
labels = []

for (i, image) in enumerate(images_train):
    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)
    label = labels_train[i]
    raw_images.append(pixels)
    features.append(hist)
    labels.append(label)

# Loop over the testing images
test_images = []
test_features = []
test_labels = []

for (i, image) in enumerate(images_test):
    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)
    label = labels_test[i]
    test_images.append(pixels)
    test_features.append(hist)
    test_labels.append(label)

# Create knn model
model = KNeighborsClassifier(n_neighbors=25, weights="distance")

# Training
model.fit(features, labels)

# Testing
accuracy = model.score(test_features, test_labels)
print("Accuracy:", accuracy)

# Save model
# file = open("knn.sk", "wb") 
# pickle.dump(model, file) 
