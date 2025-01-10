import numpy as np
from files.neural_network.layers.layer import Layer
class Perceptron:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, activation_function='relu'):
        if len(self.layers) > 0:
            input_size = self.layers[-1].weights.shape[1]
        layer = Layer(input_size, output_size, activation_function)
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = - (1/m) * np.sum(y_true * np.log(y_pred + 1e-12) + (1 - y_true) * np.log(1 - y_pred + 1e-12))
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
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = (np.sum(X, axis=1) > 5).astype(int).reshape(-1, 1)
    model = Perceptron()

    model.add_layer(input_size=10, output_size=5, activation_function='relu')
    model.add_layer(input_size=5, output_size=1, activation_function='sigmoid')

    epochs = 1000
    learning_rate = 0.01
    model.train(X, y, epochs, learning_rate)
