import numpy as np


# Class Neuron includes neuron's layers and they activation functions and weights
class Neuron:
    def __init__(self, input_size: int, bias: int=1):
        self.bias = bias
        self.weights = np.random.rand(input_size + self.bias) * 0.1

    def activate(self, inputs: np.array) -> float:
        assert inputs.shape == self.weights.shape
        z = np.dot(inputs, self.weights[:-1]) + self.weights[-1]
        return self.sigmoid(z)

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        return x * (1 - x)


# Layer include contains neurons and they activation functions
class Layer:
    def __init__(self, num_neurons: int, input_size: int):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def forward(self, inputs: float) -> np.array:
        return np.array([neuron.activate(inputs) for neuron in self.neurons])


# Contain few layers and methods for they forward and back propagation
class Network:
    def __init__(self, layers: [Layer]):
        self.layers = layers

    def forward(self, x: np.array):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backpropagation(self):
        pass
