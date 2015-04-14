from units.lstm_unit import LSTMUnit
from units.embed_unit import EmbedUnit
from utils import dataset
import numpy as np

class LSTMEncoder(object):
    """ A neural network layer composed of LSTM objects """
    def __init__(self, vocab_size, num_input, num_hidden, num_layers):
        """ Creates a neural network layer composed of LSTM units 
        which projects the number of input units into a fixed output size
        to be used for 
        @param vocab_size: vocabulary size (# of unique words into LSTM)
        @param num_input: embedding size of a single word
        @param num_hidden: number of hidden units into the LSTM
        @param num_layers: number of layers in the LSTM
        @param num_units: number of units in each layer """
        
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_layers = num_layers

        # Need an embedding model to map word indices to vectors
        self.embed_model = EmbedUnit.init(vocab_size, num_input)
        self.input_model = LSTMUnit.init(num_input, num_hidden)
        self.hidden_model = LSTMUnit.init(num_hidden, num_hidden)

    def forward(self, Xi, backprop = False):
        """ Feed forwards the input and convert it into a fixed-size feature vector 
        @param Xi: Input sequence to feedforward 
        @return: An array of hidden and cell states starting from the 0th to n-1th layer """
        input_size = len(Xi)

        # Set previous hidden/cell states to zero
        h_prev = np.squeeze(np.random.rand(self.num_hidden,1))
        h_prev[:] = 0
        c_prev = h_prev

        tot_hidden_states = []
        tot_cell_states = []
        hidden_states = []
        cell_states = []

        # Feed forward the neural network for input layer
        for i in range(0, input_size):
            x_t = Xi[i]
            
            # Get embedding representation
            input_t = EmbedUnit.forward(self.embed_model, x_t)

            # Get the new hidden and cell states
            hidden_state, cell_state = LSTMUnit.forward(self.input_model, input_t, h_prev, c_prev)

            # Add them to array (to be used in the next layer)
            hidden_states.append(hidden_state)
            cell_states.append(cell_state)

            # Go forward in the input state
            h_prev = hidden_state
            c_prev = cell_state

        tot_hidden_states.append(hidden_state)
        tot_cell_states.append(cell_state)

        # Now continue for the hidden layers
        for i in range(1,  self.num_layers):
            input_size = len(hidden_states)
            next_hidden_states = []
            next_cell_states = []
            h_prev = np.zeros_like(h_prev)
            c_prev = np.zeros_like(c_prev)

            # Feed forward the neural network for input layer
            for j in range(0, input_size):
                x_t = hidden_states[j]

                # Get the new hidden and cell states
                hidden_state, cell_state = LSTMUnit.forward(self.hidden_model, x_t, h_prev, c_prev)

                # Add them to array (to be used in the next layer)
                next_hidden_states.append(hidden_state)
                next_cell_states.append(cell_state)

            hidden_states = next_hidden_states
            cell_states = next_cell_states

            tot_hidden_states.append(np.array(hidden_state))
            tot_cell_states.append(cell_state)

        return tot_hidden_states, tot_cell_states,

    def backward(self, losses):
        """Do a backprop pass on the lstm graph """



