Part 1: K-Nearest Neighbors Classification

`euclidean_distance` and `manhattan_distance` Functions in util.py

Both functions take two numpy arrays (`x1` and `x2`) representing vectors of the same length and return the corresponding distance.

Euclidean Distance:
Euclidean distance between two vectors x1 and x2 dimensions is calculated using the following formula:

 d(x1,x2) = sqrt{sum_{i=1-n} x1i- x2i ]

Here,
- x1i and x2i are the components of vectors x1 and x2respectively.
- The summation runs over all dimensions  i of the vectors.

In the code, this formula is implemented as `np.sqrt(np.sum((x1 - x2) ** 2))

Manhattan Distance:
Manhattan distance (also known as L1 distance or Taxicab distance) between two vectors x1 and x2 dimensions is calculated using the following formula:

d(x1,x2)= sum(i=1-n) | x1i-x2i |

Here,
- x1i and x2i are the components of vectors x1 and x2respectively.
- The summation runs over all dimensions (i) of the vectors.

In the code, this formula is implemented as `np.sum(np.abs(x1 - x2)).


KNearestNeighbors Class

The `KNearestNeighbors` class is a machine learning implementation of a K-Nearest Neighbors classifier from scratch. It supports two distance metrics (`l1` for Manhattan and `l2` for Euclidean), two weight functions (`uniform` and `distance`), and allows the user to specify the number of neighbors (`n_neighbors`). The class has the following key components:

Attributes:
  - `n_neighbors`: Number of neighbors considered for predictions.
  - `weights`: Weight function for prediction ('uniform' or 'distance').
  - `_X`: Training data.
  - `_y`: True class values for training data.
  - `_distance`: Selected distance metric function.

__init__:
    - Initializes the KNN model with user-specified parameters.
    - Validates input parameters and sets up the distance metric function.

fit(X, y):
    - Fits the model to the provided training data (`X`) and true class labels (`y`).
    - Stores the training data and labels in the model.

predict(X):
    - Predicts class target values for the given test data (`X`) using the fitted model.
    - For each test sample, calculates distances to all training samples using the selected distance metric.
    - Identifies the indices of the k-nearest neighbors.
    - Utilizes the majority class of neighbors for predictions, considering optional distance weighting.

Part 2: Multilayer Perceptron Classification
    
    
The Multilayer Perceptron class represents a machine learning implementation of a Multilayer Perceptron (MLP) classifier from scratch. The MLP has one hidden layer, and its architecture is configurable through various parameters. The class supports activation functions like identity, sigmoid, tanh, and relu for the hidden layer, softmax for the output layer, and cross-entropy as the loss function.

Activation Functions:
The activation functions (identity, sigmoid, tanh, relu) are implemented as separate functions (identity, sigmoid, tanh, relu) and stored in a dictionary (activation_functions). These functions are used for the hidden layer activation.


Training Process:
* 		Initialization:
    * Weights are initialized using He initialization.
    * One-hot encoding is applied to the target class values.
    * The random seed is set for reproducibility.
* 		Forward Pass:
    * Hidden layer input and output are computed.
    * Output layer input and output are computed using softmax activation.
* 		Backward Pass (Backpropagation):
    * Loss is computed using cross-entropy.
    * Output errors are computed.
    * Hidden errors are backpropagated.
    * Weights and biases are updated using gradient descent.
* 		Training Loop:
    * Iteratively performs forward and backward passes for the specified number of iterations.
    * we are random the X and y traing data for better prediction
    * Loss history is recorded for analysis.
Prediction:
* The prediction method uses the trained model to make predictions on new data.
* The forward pass is performed, and the class with the highest probability is predicted.

The issue faced: given training set worked well on certain parameters of activation, neuron and learning but it causing overfitting and an increase in loss value for a few tried L2 regulation , and gradient descent when 1, 0.1 0.001 still could not achieve the exact results 

