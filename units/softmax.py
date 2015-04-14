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
    def backward(model, hidden, prob, Y):
        W = model['W']

        # First compute loss
        dl_dz = Y - prob

        # Then compute gradient of loss wrt weights
        gradDw = np.outer(dl_dz, hidden)
        model['dW'] += gradW

        # Then compute loss with respect to hidden activations
        dl_dh = W * dl_dz
        return dl_dh
