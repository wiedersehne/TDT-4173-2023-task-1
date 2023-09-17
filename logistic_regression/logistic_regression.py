import numpy as np
import pandas as pd

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    def __init__(self, alfa=0.0001, iterations=10000):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.alfa = alfa  # learning rate
        self.iterations = iterations  # number of iterations
        self.theta = None  # weights
        self.bias = None  # bias

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing
                m binary 0.0/1.0 labels
        """
        # TODO: Implement
        # make X and y into numpy arrays
        X = np.array(X.to_numpy())
        y = np.array(y.to_numpy())

        # initialise theta and bias
        self.theta = np.zeros(X.shape[1])
        self.bias = 0

        for i in range(self.iterations):
            # gradient descent
            z = np.dot(X, self.theta) + self.bias
            y_pred = sigmoid(z)

            # calculate gradients
            d_theta = np.dot(X.T, (y - y_pred))  # derivative of loss function wrt theta
            d_bias = np.sum(y - y_pred)  # derivative of loss function wrt bias

            # update parameters
            self.theta = self.theta + self.alfa * d_theta
            self.bias = self.bias + self.alfa * d_bias

        return self.theta, self.bias

    def predict(self, X: pd.DataFrame):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        # TODO: Implement

        return sigmoid(
            np.dot(X, self.theta) + self.bias
        )  # return probability-like predictions


# --- Some utility functions


def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true
    return correct_predictions.mean()


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * (np.log(1 - y_pred)))


def sigmoid(x):
    """
    Applies the logistic function element-wise

    Hint: highly related to cross-entropy loss

    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.

    Returns:
        Element-wise sigmoid activations of the input
    """
    return 1.0 / (1.0 + np.exp(-x))
