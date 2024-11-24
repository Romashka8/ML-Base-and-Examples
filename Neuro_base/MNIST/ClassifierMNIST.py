import numpy as np


class ClassifierMNIST:
	def __init__(self):
		self.weights = None
		self.n_classes = 10

	def fit(self, x_train: np.array, y_train: np.array) -> None:
		self.bias = np.ones((len(x_train), 1))
		x_train_with_bias = np.concatenate([bias, x_train], axis=1)

		weights_init = np.ones((self.n_classes, x_train_with_bias.shape[1]))

		for label in range(self.n_classes):
			indices = np.where(y_train == label)[0]
			xi = x_train_with_bias[indices]
			yi = np.full(indices.shape[0], 1)
			wi = np.linalg.pinv(xi.T @ xi) @ xi.T @ yi
			w[label] = wi

		self.weights = weights_init

	def predict(self, x: np.array) -> np.array:
		x_with_bias = np.concatenate([self.bias, x], axis=1)
		scores = self.weights @ x_with_bias.T
		return np.argmax(scores, axis=0)
