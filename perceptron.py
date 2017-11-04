"perceptron module"
import numpy as np

class Perceptron:
    "Main class for module"
    def __init__(self):
        # Randomizes the weights to value between -1 and 1
        self.weights = 2 * np.random.random((2, 1)) - 1

    def _sigmoid(self, _x, deriv=False):
        if deriv:
            return _x * (1 - _x)
        return 1 / (1 + np.exp(-_x))


    def predict(self, tinputs):
        "Predicts a new value based on an input"
        return self._sigmoid(np.dot(tinputs, self.weights))

    def train(self, tinputs, toutputs, iterations):
        "Trains the model and updates its weights through back propagation using gradient descent"
        for iteration in range(iterations):
            output = self.predict(tinputs)
            error = toutputs - output
            adjustment = np.dot(tinputs.T, error * self._sigmoid(output, 1))
            self.weights += adjustment
    