import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from preprocessing import get_data_from_tensorflow, get_train_data, get_test_data
from utils import plot_training, classes
from models import MLP, CNN
import matplotlib.pyplot as plt 
import numpy as np
import random

# Preprocessing
images_train, labels_train, images_test, labels_test = get_data_from_tensorflow()

# Creating models
mlp = MLP()

# Training MLP Model
mlp.train(images_train, labels_train)

# Testing MLP Model
precision, recall, accuracy, auc = mlp.test(images_test, labels_test)

_ , ax = plt.subplots(5, 1, figsize=(15, 5))

ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Value")
ax[0].set_title("Loss")

ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Value")
ax[1].set_title("Presicion")

ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Value")
ax[2].set_title("Recall")

ax[3].set_xlabel("Epoch")
ax[3].set_ylabel("Value")
ax[3].set_title("Accuracy")

ax[4].set_xlabel("Epoch")
ax[4].set_ylabel("Value")
ax[4].set_title("AUC")

ax[0].plot(mlp.epochs[1:], mlp.hist["loss"][1:], color="r")
ax[1].plot(mlp.epochs[1:], mlp.hist["precision"][1:], color="g")
ax[2].plot(mlp.epochs[1:], mlp.hist["recall"][1:], color="b")
ax[3].plot(mlp.epochs[1:], mlp.hist["categorical_accuracy"][1:], color="k")
ax[4].plot(mlp.epochs[1:], mlp.hist["auc"][1:], color="y")

plt.savefig("finalMLPmodel.png")
plt.show()

mlp.save()