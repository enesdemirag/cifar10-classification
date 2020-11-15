from preprocessing import *
from train import * 
from test import *
from models import *

# Preprocessing
train_data, labels = get_train_data(1)
test_data, _ = get_test_data()

# Creating models
mlp = create_mlp_model(0.01)

# Training
epochs, hist = train_model(mlp, train_data, labels, epochs=10)
plot_training(epochs, hist)

# Testing
result = test_model(mlp, test_data, labels)
print(result)