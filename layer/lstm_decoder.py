from units.lstm_unit import LSTMUnit
from units.embed_unit import EmbedUnit
from units.softmax import SoftmaxUnit
from utils import dataset
import numpy as np

class LSTMDecoder(object):
    """ An LSTM Decoder layer composed of LSTM objects """

    def __init__(self, num_input, num_hidden, num_layers, num_output):
        """ Creates a LSTM Language model layer which generates text given 
        an input
        @param num_input: number of input units into LSTM 
        @param num_hidden: number of hidden units into the LSTM
        @param num_output: number of outputs of the LSTM """
        
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.num_layers = num_layers

        self.input_model = LSTMUnit.init(num_input, num_hidden)
        self.hidden_model = LSTMUnit.init(num_hidden, num_hidden)
        self.output_model = SoftmaxUnit.init(num_hidden, num_output)
        
        # Need an embedding model to map word indices to vectors
        self.embed_model = EmbedUnit.init(num_output, num_input)

    def forward(self, x_init, h_init, c_init, Yi):
        """ Feed forwards the input and convert it into a readable sequence
        @param h_init: Array of hidden units from layers of encoding lstm
        @param c_init: Array of cell states from layers of encoding lstm
        @param Yi: Output/desired sequence
        @return: Predicted sequences """

        input_size = len(Yi)
        output_states = []
        hidden_states = []
        cell_states = []

        # TODO: Initial x value should be the special <START> symbol in embedding matrix
        x_t = x_init
        h_prev = h_init[0]
        c_prev = c_init[0]

        # Feed forward the neural network for input layer
        for i in range(0, input_size):
            input_t = EmbedUnit.forward(self.embed_model, x_t)

            # Get the new hidden and cell states
            hidden_state, cell_state = LSTMUnit.forward(self.input_model, input_t, h_prev, c_prev)

            # Add them to array (to be used in the next layer)
            hidden_states.append(hidden_state)
            cell_states.append(cell_state)

            # Set new input to be output of previous layer
            x_t = Yi[i]
            h_prev = hidden_state
            c_prev = cell_state


                # Now continue for the hidden layers
        for i in range(1,  self.num_layers):
            input_size = len(hidden_states)
            next_hidden_states = []
            next_cell_states = []
            h_prev = h_init[i]
            c_prev = c_init[i]

            # Feed forward the neural network for input layer
            for j in range(0, input_size):
                x_t = hidden_states[j]
                # Get the new hidden and cell states
                hidden_state, cell_state = LSTMUnit.forward(self.hidden_model, x_t, h_prev, c_prev)

                # Add them to array (to be used in the next layer)
                next_hidden_states.append(hidden_state)
                next_cell_states.append(cell_state)

                # If this is the final layer get the softmax activation
                if i == self.num_layers - 1:
                    output_state = SoftmaxUnit.forward(self.output_model, hidden_state)
                    output_states.append(output_state)

            hidden_states = next_hidden_states 
            cell_states = next_cell_states



        return output_states



