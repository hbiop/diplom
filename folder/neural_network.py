import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
# Активационные функции и их производные
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax_derivative(output):
    return output * (1 - output)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-7), axis=1))

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.output = softmax(self.z)
        return self.output

    def backward(self, d_output):
        # Производная softmax
        d_softmax = d_output * softmax_derivative(self.output)

        # Градиент весов и смещения
        self.weights_grad = self.inputs.T.dot(d_softmax)
        self.bias_grad = np.sum(d_softmax, axis=0, keepdims=True)

        # Градиент для предыдущего слоя
        d_input = d_softmax.dot(self.weights.T)
        return d_input

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)


    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, d_output):
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Прямой проход
            output = self.forward(X)

            # Вычисление ошибки
            loss = cross_entropy_loss(y, output)
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

            # Обратный проход
            self.backward(y - output)

            # Обновление весов
            for layer in self.layers:
                layer.weights += learning_rate * layer.weights_grad
                layer.bias += learning_rate * layer.bias_grad

    def predict(self, X):
        return self.forward(X)


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    np.set_printoptions(threshold=None)
    df = pd.read_csv('C:\\Users\\irina\\Downloads\\ObesityDataSet_raw_and_data_sinthetic.csv')

    # Разделение на входные данные и метку
    X = df.drop(['NObeyesdad'], axis=1)
    y = df['NObeyesdad'].values.reshape(-1, 1)

    # Преобразуем категориальные признаки в числовые
    categorical_columns = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Кодируем категориальные признаки
    ohe = OneHotEncoder(drop='first')  # Используем drop='first' чтобы избежать ловушки мультиколлинеарности
    X_cat_encoded = ohe.fit_transform(X[categorical_columns]).toarray()

    # Масштабируем числовые признаки
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X[numerical_columns])

    # Объединяем закодированные категориальные и масштабированные числовые признаки
    X_processed = np.hstack((X_num_scaled, X_cat_encoded))
    # Разделим данные на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    model = Model()
    model.add(DenseLayer(X_train.shape[1], 32))  # Входной слой
    model.add(DenseLayer(32, 16))  # Скрытый слой
    model.add(DenseLayer(16, 7))  # Выходной слой

    labels = [label[0] for label in y_train]
    lb = LabelBinarizer()
    y_true_one_hot = lb.fit_transform(labels)
    model.train(X_train, y_true_one_hot, epochs=5000, learning_rate=1.5)

