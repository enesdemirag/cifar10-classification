import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy


class MLP(object):
    def __init__(self, learning_rate):
        self.model = Sequential()

        self.model.add(Flatten(input_shape=(32, 32, 3)))
        self.model.add(Dense(units=3072, activation="relu"))
        self.model.add(Dense(units=1024, activation="relu"))
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dense(units=64, activation="relu"))
        self.model.add(Dense(units=10, activation="softmax"))

        self.model.compile(
            optimizer=RMSprop(learning_rate),
            loss=MeanSquaredError(),
            metrics=["accuracy"]
        )

    def train(self, features, labels, batch_size=16, epochs=10, shuffle=True):
        history = self.model.fit(features, labels, batch_size, epochs, shuffle=shuffle)
        self.epochs = history.epoch
        self.hist = pd.DataFrame(history.history)
        return self.epochs, self.hist

    def test(self, features, labels):
        _, self.accuracy = self.model.evaluate(features, labels, verbose=0)
        return self.accuracy

    def predict(self, features):
        prediction = self.model(features)
        return prediction


class CNN(object):
    def __init__(self, learning_rate):
        self.model = Sequential()

        self.model.add(Conv2D(filters=8, kernel_size=3, activation="relu", input_shape=(32, 32, 3)))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(units=32, activation="sigmoid"))
        self.model.add(Dense(units=10, activation="softmax"))

        self.model.compile(
            optimizer=Adam(learning_rate),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def train(self, features, labels, batch_size=16, epochs=10, shuffle=True):
        history = self.model.fit(features, labels, batch_size, epochs, shuffle=shuffle)
        self.epochs = history.epoch
        self.hist = pd.DataFrame(history.history)
        return self.epochs, self.hist

    def test(self, features, labels):
        _, self.accuracy = self.model.evaluate(features, labels, verbose=2)
        return self.accuracy

    def predict(self, features):
        prediction = self.model(features)
        return prediction

