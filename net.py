import numpy
import numpy as np


def activation(x):
    return 1 / (1 + numpy.exp(-x))


class Net:
    def __init__(self):
        self.layer = 3
        self.alpha = 0.001
        self.lamda = 0
        self.input = [785, 101]
        self.output = [100, 10]
        self.theta = [
            np.random.randn(785, 100),
            np.random.randn(101, 10)
        ]
        self.grad = [
            np.random.randn(785, 100),
            np.random.randn(101, 10)
        ]
        self.x = np.zeros(785)
        self.y = np.zeros(10)
        self.z = [
            np.zeros(100),
            np.zeros(10),
        ]
        self.a = [
            np.zeros(101),
            np.zeros(10),
        ]

    def forward(self):
        self.z[0] = self.x.dot(self.theta[0])
        for i in range(self.layer - 2):
            self.a[i] = numpy.insert(activation(self.z[i]), 0, 1, 0)
            self.z[i + 1] = self.a[i].dot(self.theta[i + 1])
        self.a[self.layer - 2] = activation(self.z[self.layer - 2])

    def backward(self):
        dlt = (self.a[self.layer - 2] - self.y).reshape(1, 10)
        for i in range(self.layer - 2):
            self.grad[self.layer - 2 - i] += self.a[self.layer - 3 - i].reshape(101, 1).dot(dlt)
            self.grad[self.layer - 2 - i] += self.lamda * self.theta[self.layer - 2 - i]
            dlt = np.delete(dlt.dot(self.theta[self.layer - 2 - i].T), 0, 1)
        self.grad[0] += self.x.reshape(785, 1).dot(dlt)
        self.grad[0] -= self.lamda * self.theta[0]

    def update(self, m):
        for i in range(self.layer - 1):
            self.theta[i] += self.alpha / m * self.grad[i]

    def set_zero(self):
        self.grad = [
            np.random.randn(785, 100),
            np.random.randn(101, 10)
        ]
