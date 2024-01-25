import numpy as np

# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: Niveditha Bommanahally Parameshwarappa(nibomm),Sudeepthi Rebbalapalli(surebbal), Samyak Kashyap Shah(shahsamy)
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden=16, hidden_activation='sigmoid', n_iterations=1000, learning_rate=0.01):
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    # def clip(self, value, epsilon=1e-7):
    #     return np.clip(value, epsilon, 1 - epsilon)
    # def clip_gradients(grad, threshold):
    #     return np.clip(grad, -threshold, threshold)

    def _initialize(self, X, y):
        self._X = X
        self._y = one_hot_encoding(y)

        np.random.seed(42)

        # He initialization
        self._h_weights = np.random.randn(X.shape[1], self.n_hidden) * np.sqrt(2 / self.n_hidden)
        self._h_bias = np.zeros((1, self.n_hidden))
        self._o_weights = np.random.randn(self.n_hidden, len(set(y))) * np.sqrt(2 / len(set(y)))
        self._o_bias = np.zeros((1, len(set(y))))

    def forward_pass(self, X):
        hlayer_input = np.dot(X, self._h_weights) + self._h_bias
        hlayer_output = self.hidden_activation(hlayer_input)

        outlayer_input = np.dot(hlayer_output, self._o_weights) + self._o_bias
        outlayer_output = self._output_activation(outlayer_input)

        return hlayer_input, hlayer_output, outlayer_input, outlayer_output

    def backward_pass(self, X, y, hlayer_output, outlayer_output):
        loss = self._loss_function(self._y, outlayer_output)

        output_error = outlayer_output
        output_error[range(len(X)), y] -= 1
        output_error /= len(X)

        hidden_error = np.dot(output_error, self._o_weights.T) * hlayer_output * (1 - hlayer_output)
        threshold=0.1
        hidden_error = np.clip(hidden_error, -threshold, threshold) # Adjust the threshold as needed

        self._o_weights -= self.learning_rate * np.dot(hlayer_output.T, output_error)
        self._o_bias -= self.learning_rate * np.sum(output_error, axis=0, keepdims=True)
        self._h_weights -= self.learning_rate * np.dot(X.T, hidden_error)
        self._h_bias -= self.learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

        return loss

    def fit(self, X, y):
        self._initialize(X, y)

        for epoch in range(self.n_iterations):
            combined_data = list(zip(X, y))
            # Shuffle the combined data and Unpack the shuffled data
            np.random.shuffle(combined_data)
            X_train_shuffled, y_train_shuffled = zip(*combined_data)
            # Convert back to numpy arrays if needed
            X_train_shuffled = np.array(X_train_shuffled)
            y_train_shuffled = np.array(y_train_shuffled)
            hlayer_input, hlayer_output, outlayer_input, outlayer_output = self.forward_pass(X_train_shuffled)
            loss = self.backward_pass(X_train_shuffled, y_train_shuffled, hlayer_output, outlayer_output)

            if epoch % 20 == 0:
                self._loss_history.append(loss)
                # print(f'Epoch {epoch}, Loss: {loss}')
        # print (self._o_weights, self._h_weights)        

    def predict(self, X):
        hlayer_input, hlayer_output, outlayer_input, outlayer_output = self.forward_pass(X)
        return np.argmax(outlayer_output, axis=1)