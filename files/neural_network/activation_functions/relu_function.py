import numpy as np

def relu_function(x):
    return np.maximum(x, 0)

def relu_deriative(x):
    return (x > 0).astype(int)