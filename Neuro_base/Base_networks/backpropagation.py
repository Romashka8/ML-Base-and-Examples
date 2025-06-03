import numpy as np

# ------------------------------------------------------------------------------------

activation_func = {

	'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
	'tanh': lambda x: np.tanh(x),
	'relu': lambda x: np.maximum(0, x)
}

activation_der = {

	'sigmoid': lambda x: x * (1 - x),
	'tanh': lambda x: 1 - np.tanh(x) ** 2,
	'relu': lambda x: np.where(x > 0, 1, 0)
}

# ------------------------------------------------------------------------------------

class Neuron:

	def __init__(self, input_size, func='sigmoid'):

		self.weights = np.random.randn(input_size + 1) * 0.1
		self.activation = activation_func[func]
		self.der = activation_der[func]
		self.output = 0

	def activate(self, x):

		z = np.dot(x, self.weights[:-1]) + self.weights[-1]
		self.output = self.activation(z)

		return self.output

# ------------------------------------------------------------------------------------

class Layer:

	def __init__(self, num_neurons, input_size, func='sigmoid'):
		
		self.neurons = [Neuron(input_size, func) for _ in range(num_neurons)]
		self.output = None
		self.error = None
		self.delta = None

	def forward(self, x):
		
		self.output = np.array([neuron.activate(x) for neuron in self.neurons])
		return self.output

# ------------------------------------------------------------------------------------

class Network:

	def __init__(self, layers):
		
		self.layers = layers

	def forward(self, x):

		for layer in self.layers: x = layer.forward(x)
		
		return x

	def backward(self, x, y, learning_rate):
		
		output = self.forward(x)

		error = y - output
		self.layers[-1].error = error
		self.layers[-1].delta = error * self.layers[-1].neurons[0].der(output)

		for i in reversed(range(len(self.layers) - 1)):

			layer = self.layers[i]
			next_layer = self.layers[i + 1]

			layer.error = np.dot(next_layer.delta, np.array([neuron.weights[:-1] for neuron in next_layer.neurons]))
			layer.delta = layer.error * np.array([neuron.der(neuron.output) for neuron in layer.neurons])

		for i in range(len(self.layers)):
			
			layer = self.layers[i]
			inputs = x if i == 0 else self.layers[i - 1].output

			for j, neuron in enumerate(layer.neurons):

				for k in range(len(neuron.weights) - 1):

					neuron.weights[k] += learning_rate * layer.delta[j] * inputs[k]

				neuron.weights[-1] += learning_rate * layer.delta[j]

	def train(self, x, y, learning_rate, epochs):

		for epoch in range(epochs):

			for xi, yi in zip(x, y):

				self.backward(xi, yi, learning_rate)


# ------------------------------------------------------------------------------------

if __name__ == '__main__':

	input_size = 3
	hidden_size = 5
	output_size = 1

	layer1 = Layer(hidden_size, input_size, func='relu')
	layer2 = Layer(output_size, hidden_size, func='sigmoid')

	network = Network([layer1, layer2])

	X = np.array([[0.5, 0.1, 0.4], [0.9, 0.7, 0.3], [0.2, 0.8, 0.6]])
	y = np.array([[1], [0], [1]])

	learning_rate = 0.1
	epochs = 10000

	network.train(X, y, learning_rate, epochs)

	for xi in X:
	    output = network.forward(xi)
	    print("Input:", xi, "Output:", output)
