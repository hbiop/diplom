import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation_function='relu'):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation_function = activation_function

    def activate(self, x):
        if self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        return x

    def activate_derivative(self, x):
        if self.activation_function == 'relu':
            return np.where(x <= 0, 0, 1)
        elif self.activation_function == 'sigmoid':
            return self.activate(x) * (1 - self.activate(x))
        elif self.activation_function == 'tanh':
            return 1 - np.tanh(x) ** 2
        return np.ones_like(x)

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.inputs, self.weights) + self.biases
        self.outputs = self.activate(self.z)
        return self.outputs

    def backward(self, dA):
        dZ = dA * self.activate_derivative(self.z)
        m = self.inputs.shape[0]
        self.dW = (1 / m) * np.dot(self.inputs.T, dZ)
        self.db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, self.weights.T)
        return dA_prev