import numpy as np

from files.data_preprocessing.data_cleaning.data_service import DataService
from files.neural_network.layers.layer import Layer
class Perceptron:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, activation_function='relu'):
        print(f"nach = {input_size}", sep="\n")
        if len(self.layers) > 0:
            input_size = self.layers[-1].weights.shape[1]
        print(f"kon = {input_size}", sep="\n")
        layer = Layer(input_size, output_size, activation_function)
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def compute_loss(self, y_true, y_pred):
        loss = np.mean(np.square(y_true - y_pred))
        return loss

    def backward(self, y_true, y_pred):
        m = y_true.shape[0]
        dA = - (y_true / (y_pred + 1e-12)) + (1 - y_true) / (1 - y_pred + 1e-12)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dW
            layer.biases -= learning_rate * layer.db

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(y, y_pred)
            self.update(learning_rate)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')


if __name__ == '__main__':
    d = DataService()
    d.read_data("D:\\конференция\\Iris.csv")
    d.prepare_data()
    x_train, x_text = d.split_data()

    # Преобразуем целевую переменную в одномерный массив
    target_column = "Species"
    y_train = x_train[target_column].values.ravel()




    model = Perceptron()

    model.add_layer(input_size=6, output_size=20, activation_function='relu')
    model.add_layer(input_size=20, output_size=5, activation_function='relu')
    model.add_layer(input_size=20, output_size=5, activation_function='relu')
    model.add_layer(input_size=20, output_size=5, activation_function='relu')
    model.add_layer(input_size=20, output_size=5, activation_function='relu')
    model.add_layer(input_size=20, output_size=5, activation_function='relu')
    model.add_layer(input_size=3, output_size=120, activation_function='sigmoid')

    epochs = 1000
    learning_rate = 0.01
    print(x_train.shape)
    print(y_train.shape)

    # Передаем одномерный массив y_train
    model.train(x_train, y_train, epochs, learning_rate)