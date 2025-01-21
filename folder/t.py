import pickle
from folder.a import NeuralNetwork, Layer, SigmoidFunction
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from files.data_preprocessing.data_cleaning.data_service import DataService

if __name__ == "__main__":
    with open('D:\Acollege\mydict.pkl', 'rb') as f:
        nn = pickle.load(f)

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
    sum = 0
    t = np.array(x_test)
    for i in range(len(t)):
     output = nn.forward(t[i:i + 1])
     predicted_classes = np.argmax(output)
     true_classes = np.argmax(Y_test_ar[i])
     sum += predicted_classes == true_classes
     print(true_classes)
     print(predicted_classes)
     accuracy = sum / len(Y_test_ar)

    # print(f"accuracy is {accuracy}")