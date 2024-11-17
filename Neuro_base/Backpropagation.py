# https://habr.com/ru/companies/otus/articles/816667/
import numpy as np


# активационные функции и их производные
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# класс Neuron
class Neuron:
    def __init__(self, input_size, activation='sigmoid'):
        self.weights = np.random.randn(input_size + 1) * 0.1  # +1 для смещения (bias)
        self.activation_function = self._get_activation_function(activation)
        self.activation_derivative = self._get_activation_derivative(activation)
        self.output = None

    def _get_activation_function(self, activation):
        if activation == 'sigmoid':
            return sigmoid
        elif activation == 'tanh':
            return tanh
        elif activation == 'relu':
            return relu

    def _get_activation_derivative(self, activation):
        if activation == 'sigmoid':
            return sigmoid_derivative
        elif activation == 'tanh':
            return tanh_derivative
        elif activation == 'relu':
            return relu_derivative

    def activate(self, inputs):
        z = np.dot(inputs, self.weights[:-1]) + self.weights[-1]
        self.output = self.activation_function(z)
        return self.output


# класс Layer
class Layer:
    def __init__(self, num_neurons, input_size, activation='sigmoid'):
        self.neurons = [Neuron(input_size, activation) for _ in range(num_neurons)]
        self.output = None
        self.error = None
        self.delta = None

    def forward(self, inputs):
        self.output = np.array([neuron.activate(inputs) for neuron in self.neurons])
        return self.output


# класс Network
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y, learning_rate):
        # прямое распространение для получения выходных значений
        output = self.forward(X)

        # вычисление ошибки на выходном слое
        error = y - output
        self.layers[-1].error = error
        self.layers[-1].delta = error * self.layers[-1].neurons[0].activation_derivative(output)

        # передача ошибки обратно через слои
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            layer.error = np.dot(next_layer.delta, np.array([neuron.weights[:-1] for neuron in next_layer.neurons]))
            layer.delta = layer.error * np.array(
                [neuron.activation_derivative(neuron.output) for neuron in layer.neurons])

        # обновление весов
        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = X if i == 0 else self.layers[i - 1].output
            for j, neuron in enumerate(layer.neurons):
                for k in range(len(neuron.weights) - 1):
                    neuron.weights[k] += learning_rate * layer.delta[j] * inputs[k]
                neuron.weights[-1] += learning_rate * layer.delta[j]  # Обновление смещения

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                self.backward(xi, yi, learning_rate)


# создание и обучение сети
input_size = 3
hidden_size = 5
output_size = 1

layer1 = Layer(hidden_size, input_size, activation='relu')
layer2 = Layer(output_size, hidden_size, activation='sigmoid')

network = Network([layer1, layer2])

# пример данных для тренировки
X = np.array([[0.5, 0.1, 0.4], [0.9, 0.7, 0.3], [0.2, 0.8, 0.6]])
y = np.array([[1], [0], [1]])

# параметры обучения
learning_rate = 0.1
epochs = 10000

# обучение сети
network.train(X, y, learning_rate, epochs)

# тестирование сети
for xi in X:
    output = network.forward(xi)
    print("Input:", xi, "Output:", output)
    
