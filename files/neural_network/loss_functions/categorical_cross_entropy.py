import numpy as np

def categorical_cross_entropy(x):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    cce = -np.sum(targets * np.log(predictions)) / predictions.shape[0]
    return cce

def categorical_cross_entropy_derivative(predictions, targets):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    derivatives = -targets / predictions
    return derivatives