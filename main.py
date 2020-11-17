from preprocessing import *
from models import *

# Preprocessing
images_train, labels_train, images_test, labels_test = get_data_from_tensorflow()

# Creating models
mlp = MLP(0.01)
cnn = CNN(0.01)

# Training MLP Model
mlp.train(images_train, labels_train, epochs=10)
plot_training(mlp)

# Testing MLP Model
accuracy_mlp = mlp.test(images_test, labels_test)
print(accuracy_mlp)

# Training CNN Model
cnn.train(images_train, labels_train, epochs=10)
plot_training(cnn)

# Testing CNN Model
accuracy_cnn = cnn.test(images_test, labels_test)
print(accuracy_cnn)

# Prediction
# y1 = predict(mlp, img)
# y2 = predict(cnn, img)
# print(y1, y2)