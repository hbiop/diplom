import numpy as np

def softmax_function(x):
    sum = np.exp(x - np.max(x))
    return sum / np.sum(sum)

def softmax_deriative(x):
    n = len(x)

    J = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                J[i][j] = s[i] * (1 - s[i])
            else:
                J[i][j] = -s[i] * s[j]

    return J