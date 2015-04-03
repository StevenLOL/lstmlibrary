from networks.lstm_encoder_decoder import LSTMEncoderDecoder

from utils import dataset
import numpy as np

# Load dataset
params = dataset.load_dataset("input_dummy.txt", "output_dummy.txt")

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

# res, error = encoder_decoder.forward(Xi, Yi)
predictions = encoder_decoder.predict(Xi, beam_size = 3)

for predList in predictions:
    logProb = predList[0]
    prediction = predList[1]
    print("Log probability is: {0}".format(logProb))
    print("Prediction is: {0}".format(prediction))



def clean_files(dir_name):
    import os
    directory = os.listdir(dir_name)
    for filename in directory:
        if filename[-3:] == 'pyc':
            print '- ' + filename
            os.remove("{0}/{1}".format(dir_name,filename))

clean_files('./layer')
clean_files('./networks')
clean_files('.')
clean_files('./units')
clean_files('./utils')