import pandas as pd
from datetime import datetime as dt
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.utils import plot_model


class MLP(object):
    def __init__(self):
        self.model = Sequential()

        self.model.add(Flatten(input_shape=(32, 32, 3)))
        self.model.add(Dense(units=1024, activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=512, activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=10,  activation="softmax"))

        self.model.compile(
            optimizer = "adam",
            loss      = "categorical_crossentropy",
            metrics   = ["accuracy"]
        )

    def train(self, features, labels, batch_size=32, epochs=50, shuffle=True):
        history     = self.model.fit(features, labels, batch_size, epochs, shuffle=shuffle)
        self.epochs = history.epoch
        self.hist   = pd.DataFrame(history.history)
        return self.epochs, self.hist

    def test(self, features, labels):
        _, self.accuracy = self.model.evaluate(features, labels, verbose=0)
        return self.accuracy

    def predict(self, img):
        return self.model.predict(img)
    
    def save(self, path="./saved_models/"):
        timestamp = dt.timestamp(dt.now())
        filename = path + "MLP_" + str(timestamp)
        
        plot_model(self.model, to_file=filename + ".png", show_shapes=True, show_layer_names=True)
        self.model.save(filename + ".h5")


class CNN(object):
    def __init__(self):
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
            optimizer = "adam",
            loss      = "categorical_crossentropy",
            metrics   = ["accuracy"]
        )

    def train(self, features, labels, batch_size=16, epochs=10, shuffle=True):
        history     = self.model.fit(features, labels, batch_size, epochs, shuffle=shuffle)
        self.epochs = history.epoch
        self.hist   = pd.DataFrame(history.history)
        return self.epochs, self.hist

    def test(self, features, labels):
        _, self.accuracy = self.model.evaluate(features, labels, verbose=2)
        return self.accuracy

    def predict(self, img):
        return self.model.predict(img)

    def save(self, path="./saved_models/"):
        timestamp = dt.timestamp(dt.now())
        filename = path + "CNN_" + str(timestamp)
        
        plot_model(self.model, to_file=filename + ".png", show_shapes=True, show_layer_names=True)
        self.model.save(filename + ".h5")