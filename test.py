import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from preprocessing import get_data_from_tensorflow
from utils import load_model, classes
from models import MLP
import matplotlib.pyplot as plt 
import numpy as np
import random

# Preprocessing
_, _, images_test, labels_test = get_data_from_tensorflow()

# Creating models
model = load_model("saved_models/finalMLPmodel.h5")
# print(model.summary())

# Testing MLP Model
# loss, precision, recall, accuracy, auc = model.evaluate(images_test, labels_test, verbose=0)

# Prediction MLP Model	
_, ax = plt.subplots(3, 4, figsize=(10, 10))
ax = ax.ravel()

for i in range(12):
    rand_sample = random.randint(0,1000)
    img = images_test[rand_sample].reshape((1, 32, 32, 3))	
    y_orig = labels_test[rand_sample]
    y_pred = model.predict(img)

    y_orig = list(y_orig).index(1)
    y_pred = [1 if i == max(y_pred[0]) else 0 for i in y_pred[0]].index(1)
    
    ax[i].axis('off')
    ax[i].imshow(img[0])	
    ax[i].set_title("True: %s \nPredict: %s" % (classes[y_orig], classes[y_pred]))	
plt.show()
