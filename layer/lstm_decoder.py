from units.lstm_decoder_unit import LSTMDecoderUnit
from units.embed_unit import EmbedUnit
from units.softmax import SoftmaxUnit
from utils import dataset
import numpy as np

class LSTMDecoder(object):
    """ An LSTM Decoder layer composed of LSTM objects """

    def __init__(self, num_input, num_hidden, num_output):
        """ Creates a LSTM Language model layer which generates text given 
        an input
        @param num_input: number of input units into LSTM 
        @param num_hidden: number of hidden units into the LSTM
        @param num_output: number of outputs of the LSTM """
        
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.input_model = LSTMDecoderUnit.init(num_input, num_hidden, num_output)
        self.output_model = SoftmaxUnit.init(num_hidden, num_output)
        
        # Need an embedding model to map word indices to vectors
        self.embed_model = EmbedUnit.init(num_output, num_input)

    def forward(self, x_init, h_init, Yi):
        """ Feed forwards the input and convert it into a fixed-size feature vector 
        @param h_init: First hidden input into the LSTM
        @param Yi: Output/desired sequence
        @return: Predicted sequences """

        input_size = len(Yi)
        output_states = []
        hidden_states = []
        cell_states = []

        # Initial x value should be the special <START> symbol in embedding matrix
        x_t = x_init
        h_prev = h_init

                # Set previous hidden/cell states to zero
        c_prev = np.squeeze(np.random.rand(self.num_hidden,1))
        c_prev[:] = 0

        # Feed forward the neural network for input layer
        for i in range(0, input_size):
            input_t = EmbedUnit.forward(self.embed_model, x_t)

            # Get the new hidden and cell states
            hidden_state, cell_state = LSTMDecoderUnit.forward(self.input_model, input_t, h_prev, c_prev)

            # Get the softmax activations
            output_state = SoftmaxUnit.forward(self.output_model, hidden_state)

            # Add them to array (to be used in the next layer)
            hidden_states.append(hidden_state)
            cell_states.append(cell_state)

            output_states.append(output_state)

            # Set new input to be output of previous layer
            x_t = Yi[i]
            h_prev = hidden_state
            c_prev = cell_state

        return output_states



