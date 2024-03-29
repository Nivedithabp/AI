# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by:Niveditha Bommanahally Parameshwarappa(nibomm),Sudeepthi Rebbalapalli(surebbal), Samyak Kashyap Shah(shahsamy)
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = y

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        predictions = []
        print(self.weights, self.n_neighbors)
        for sample in X:
            # Calculate distances between the current sample and all samples in the training data
            distances = [self._distance(sample, x) for x in self._X]

            # Get indices of the k-nearest neighbors
            indices = np.argsort(distances)[:self.n_neighbors]

            # Get the corresponding target values of the k-nearest neighbors
            neighbors_labels = self._y[indices]

            # Make predictions based on the majority class of the neighbors
            if self.weights == 'uniform':
                prediction = np.bincount(neighbors_labels).argmax()
            elif self.weights == 'distance':
                weights = 1 / (np.array(distances)[indices] + 1e-6)  # Avoid division by zero
                prediction = np.bincount(neighbors_labels, weights).argmax()

            predictions.append(prediction)

        return np.array(predictions)