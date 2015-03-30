from layer.lstm_encoder import LSTMEncoder
from layer.lstm_decoder import LSTMDecoder
from utils import dataset
import numpy as np

input_vocab_size = 50
output_vocab_size = 2
num_input = 100
num_hidden = 1000
num_layers = 4

num_new_input = 50

Xi = []
for i in range(0, 3):
    x = np.squeeze(np.random.rand(input_vocab_size, 1))
    Xi.append(x)

Yi = []
for i in range(0, 3):
    y = np.squeeze(np.random.rand(output_vocab_size, 1))
    Yi.append(y)

x_init = np.squeeze(np.random.rand(output_vocab_size, 1))

encodingLayer = LSTMEncoder(input_vocab_size, num_input, num_hidden, num_layers)
h_init = encodingLayer.forward(Xi)

decodingLayer = LSTMDecoder(5, num_hidden * num_layers, output_vocab_size)
res = decodingLayer.forward(x_init, h_init, Yi)

print("Size of results: {0}".format(len(res)))
print("DONE")


