from utils import functions
import numpy as np

class EmbedUnit():
    @staticmethod
    def init(input_size, output_size):
        W = functions.initw(input_size, output_size)
        model = {'W' : W}
        return model

    @staticmethod
    def forward(model, input):
        W = model['W']
        prod = input.dot(W)
        return prod

    @staticmethod
    def backward(model, input, diff):
        # TODO
        return deltaW
