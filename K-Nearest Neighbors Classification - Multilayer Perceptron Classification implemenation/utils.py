# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by:Niveditha Bommanahally Parameshwarappa(nibomm),Sudeepthi Rebbalapalli(surebbal), Samyak Kashyap Shah(shahsamy)
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff

import numpy as np

def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    return np.sum(np.abs(x1 - x2))


def identity(x, derivative=False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if derivative:
        return np.ones_like(x)
    else:
        return x


def sigmoid(x, derivative=False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    s = 1 / (1 + np.exp(-x))
    if derivative:
        return s * (1 - s)
    else:
        return s


def tanh(x, derivative=False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if derivative:
        return 1 - np.tanh(x) ** 2
    else:
        return np.tanh(x)


def relu(x, derivative=False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if derivative:
        return np.where(x > 0, 1, 0)
    else:
        return np.maximum(0, x)


def softmax(x, derivative=False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - c)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        s = softmax(x)
        return s * (1 - s)


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """
    return -np.sum(y * np.log(p + 1e-15)) / len(y)


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicates the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    n_classes = len(np.unique(y))
    one_hot_encoded = np.zeros((len(y), n_classes))
    one_hot_encoded[np.arange(len(y)), y] = 1
    return one_hot_encoded
