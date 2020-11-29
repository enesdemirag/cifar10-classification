from preprocessing import get_data_from_tensorflow, get_train_data, get_test_data
from utils import save_model, plot_training
from models import MLP, CNN

# Preprocessing
# images_train, labels_train, images_test, labels_test = get_data_from_tensorflow()
images_train, labels_train = get_train_data(1)
images_test, labels_test = get_test_data()

# Creating models
mlp = MLP(0.01)
cnn = CNN(0.01)

# Training MLP Model
mlp.train(images_train, labels_train, epochs=1)
plot_training(mlp)
save_model(mlp, "mlp")
exit()
# Testing MLP Model
accuracy_mlp = mlp.test(images_test, labels_test)
print(accuracy_mlp)

# Training CNN Model
cnn.train(images_train, labels_train, epochs=1)
plot_training(cnn)
save_model(cnn, "cnn")

# Testing CNN Model
accuracy_cnn = cnn.test(images_test, labels_test)
print(accuracy_cnn)

# Prediction
# y1 = predict(mlp, img)
# y2 = predict(cnn, img)
# print(y1, y2)
