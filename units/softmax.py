from utils import functions
import numpy as np

class SoftmaxUnit():
    @staticmethod
    def init(input_size, output_size):
        W = functions.initw(input_size, output_size)
        model = {'W' : W}
        return model

    @staticmethod
    def forward(model, input):
        W = model['W']
        prod = np.exp(input.dot(W))
        output = prod / sum(prod)
        return output

    @staticmethod
    def backward(model, input, diff):
        W = model['W']
        deltaW = np.zeros_like(W)
        for j in range(0, len(diff)):
            for i in range(0, len(input)):
                res = -input[i] * diff[j]
                deltaW[i][j] += res
        return deltaW
