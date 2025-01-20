import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from  files.data_preprocessing.data_cleaning.data_service import DataService
from files.neural_network.layers.layer import Layer

class NeuralNetwork:
    def __init__(self):
        self.__init__()
        self.layers = []

    #def add_layer:


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

    def train(self, X, y, epochs, learning_rate, batch_size=32, verbose=True):
        num_batches = int(np.ceil(X.shape[0] / batch_size))
        for epoch in range(epochs):
            print(epoch)
            shuffled_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, X.shape[0])
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_batch, y_pred)
                self.backward(y_batch, y_pred)
                self.update(learning_rate)
            if verbose and (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

if __name__ == '__main__':
    d = DataService()
    d.read_data("D:\\конференция\\Iris.csv")
    d.prepare_data()
    x_train, x_text = d.split_data()

    # Преобразуем целевую переменную в одномерный массив
    target_column = "Species"
    y_train = x_train[target_column].values.ravel()

    # Нормализуем данные
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # Разделяем данные на тренировочные и тестовые
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    model = Perceptron()

    model.add_layer(input_size=x_train.shape[1], output_size=64, activation_function='relu')
    model.add_layer(input_size=64, output_size=32, activation_function='relu')
    model.add_layer(input_size=32, output_size=32, activation_function='softmax')

    epochs = 1000
    learning_rate = 0.001
    batch_size = 32

    model.train(x_train, y_train, epochs, learning_rate, batch_size)

    # Оценка на валидационных данных
    y_val_pred = model.predict(x_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Валидационная точность: {val_accuracy:.4f}")