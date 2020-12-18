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
mlp = MLP(lr=0.03)

# Training MLP Model
mlp.train(images_train, labels_train, epochs=100)

# Prediction MLP Model
_, ax = plt.subplots(1, 5)
for i in range(5):
    img = images_test[random.randint(0,1000)].reshape((1, 32, 32, 3))
    y = mlp.predict(img)
    y = np.where(y[0])[0][0]
    y = classes[y]
    ax[i].imshow(img[0])
    ax[i].set_title(y)
plt.show()

# Testing MLP Model
accuracy_mlp = mlp.test(images_test, labels_test)
print(accuracy_mlp)

plot_training(mlp)
mlp.save()