from abc import ABC

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from files.data_preprocessing.data_cleaning.data_service import DataService


class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        if activation_function == "relu":
            self.activation_function = ReluFunction()
        elif activation_function == "softmax":
            self.activation_function = SoftMaxFunction()
        elif activation_function == "sigmoid":
            self.activation_function = SigmoidFunction()

    def forward(self, input_data):
        self.input_data = input_data
        self.z = np.dot(self.input_data, self.weights) + self.biases
        return self.activation_function.function(self.z)

    def backward(self, output_gradient, learning_rate):
        activation_gradient = self.activation_function.backward(self.z)
        grad_z = output_gradient * activation_gradient
        self.weights -= np.dot(self.input_data.T, grad_z) * learning_rate
        self.biases -= np.sum(grad_z, axis=0, keepdims=True) * learning_rate
        return np.dot(grad_z, self.weights.T)

class ReluFunction:
    @staticmethod
    def function(x):
        return np.maximum(0, x)

    @staticmethod
    def backward(x):
        return (x > 0).astype(float)

class SoftMaxFunction:
    @staticmethod
    def function(x):
        exp_x = np.exp(x - np.max(x))  # Устраняем переполнение экспоненциальной функции
        return exp_x / np.sum(exp_x)

    @staticmethod
    def backward(x):
        p = SoftMaxFunction.function(x)
        n = len(p)
        jacobian_matrix = -np.outer(p, p)
        np.fill_diagonal(jacobian_matrix, p * (1 - p))
        return jacobian_matrix

class SigmoidFunction:
    @staticmethod
    def function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        sig = SigmoidFunction.function(x)
        return sig * (1 - sig)

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, activation_function):
        layer = Layer(input_size, output_size, activation_function)
        self.layers.append(layer)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                output = self.forward(X[i:i+1])
                loss = self.loss(y[i:i+1], output)
                print(f'Epoch {epoch + 1}/{epochs}, Sample {i + 1}/{X.shape[0]}, Loss: {loss}')
                output_gradient = self.loss_derivative(y[i:i+1], output)
                self.backward(output_gradient, learning_rate)

    @staticmethod
    def loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)  # MSE

    @staticmethod
    def loss_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


def calculate_accuracy(predictions, true_labels):
    # Получаем индексы классов с максимальными вероятностями
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)  # Если метки закодированы в one-hot формате

    # Считаем количество правильных предсказаний
    correct_predictions = np.sum(predicted_classes == true_classes)

    # Вычисляем точность
    accuracy = correct_predictions / len(true_labels)
    return accuracy


if __name__ == "__main__":
    ds = DataService()
    ds.read_data('C:\\Users\\irina\\Downloads\\ObesityDataSet_raw_and_data_sinthetic.csv')
    ds.prepare_data()
    x_train, x_test, y_train, y_test = ds.split_data()
    print(x_train)
    encoder = OneHotEncoder()
    encoded_categories = encoder.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
    encoded_categories_test = encoder.fit_transform(np.array(y_test).reshape(-1, 1)).toarray()

    X = np.array(x_train)
    y = np.array(encoded_categories)
    X_test = np.array(x_test)
    Y_test_ar = np.array(encoded_categories_test)
    nn = NeuralNetwork()
    nn.add_layer(input_size=16, output_size=20, activation_function="sigmoid")
    nn.add_layer(input_size=20, output_size=40, activation_function="sigmoid")
    nn.add_layer(input_size=40, output_size=40, activation_function="sigmoid")
    nn.add_layer(input_size=40, output_size=7, activation_function="sigmoid")
    sum = 0
    nn.train(X, y, epochs=8000, learning_rate=0.05)
    t = np.array(x_test)
    for i in range(len(t)):
        output = nn.forward(t[i:i + 1])
        predicted_classes = np.argmax(output)
        true_classes = np.argmax(Y_test_ar[i])
        sum += predicted_classes == true_classes
        print(true_classes)
        print(predicted_classes)
    accuracy = sum / len(Y_test_ar)

    print(f"accuracy is {accuracy}")


