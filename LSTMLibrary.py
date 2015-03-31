from networks.lstm_encoder_decoder import LSTMEncoderDecoder

from utils import dataset
import numpy as np

input_vocab_size = 50
output_vocab_size = 100
input_embed_size = 100
num_input = 100
num_memory_units = 1000
num_layers = 4


num_new_input = 50

Xi = range(11, 17)
Yi = range(5, 12)

encoder_decoder = LSTMEncoderDecoder(input_vocab_size, output_vocab_size, input_embed_size, input_embed_size,
                                     num_layers, num_memory_units)

res, error = encoder_decoder.forward(Xi, Yi)

print("Size of results: {0}".format(len(res)))
print("DONE")


