import numpy as np
from MLP import CustomMLP
import keras


def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten = True)

y_t = np.zeros((y_train.size, 10))
y_t[np.arange(y_train.size), y_train] = 1

        
k = CustomMLP()
    
k.train(X_train, y_t, epochs = 3)

k.save_model()