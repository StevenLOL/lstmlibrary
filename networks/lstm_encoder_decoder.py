from layer.lstm_decoder import LSTMDecoder
from layer.lstm_encoder import LSTMEncoder
from utils import dataset
import numpy as np

class LSTMEncoderDecoder(object):
    """ An LSTM Encoder Decoder Network as described in the sequence
   to sequence paper """

    def __init__(self, input_vocab_size, output_vocab_size, input_embed_size, output_embed_size, num_layers,
                 num_memory_units):
        """ Creates a LSTM Language model layer which generates text given 
        an input
        @param num_input: number of input units into LSTM 
        @param num_hidden: number of hidden units into the LSTM
        @param num_output: number of outputs of the LSTM """

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.input_embed_size = input_embed_size
        self.output_embed_size = output_embed_size
        self.num_layers = num_layers
        self.num_hidden_units = num_memory_units

        self.encodingLayer = LSTMEncoder(self.input_vocab_size, self.input_embed_size, 
                                         self.num_hidden_units, self.num_layers)
        self.decodingLayer = LSTMDecoder(output_embed_size, self.num_hidden_units, num_layers, output_vocab_size)

    def forward(self, Xi, Yi):
        """ Feed forwards the input sequence, returns predicted sequence and 
        softmax error 
        @param Xi: Input sequence
        @param Yi: Output/desired sequence
        @return: Predicted sequences, Error """

        h_out, c_out = self.encodingLayer.forward(Xi)
        res = self.decodingLayer.forward(0, h_out, c_out, Yi)

        # Accumulate softmax error
        totError = 0.0
        for i in range(0, len(res)):
            # Get expected word index
            expectedIndex = Yi[i]

            # Get current softmax probability
            probDistr = res[i]

            # Get the - log probability
            logProb = -np.log(probDistr[expectedIndex])
            totError += logProb

        return res, totError

    def predict(self, Xi, beam_size = 1):
        """ Generates a sequence for given input and beam size 
        @param input: Input sequence to convert
        @param beam_size: Beam size for decoding
        @return list of predicted indeces
        """
        
        h_out, c_out = self.encodingLayer.forward(Xi)
        res = self.decodingLayer.predict(0, h_out, c_out, beam_size)
        return res





