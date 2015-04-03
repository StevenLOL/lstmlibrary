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
        for i in range(1, self.num_layers):
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

                # Reset input and hidden
                h_prev = hidden_state
                c_prev = cell_state

                # If this is the final layer get the softmax activation
                if i == self.num_layers - 1:
                    output_state = SoftmaxUnit.forward(self.output_model, hidden_state)
                    output_states.append(output_state)

            hidden_states = next_hidden_states 
            cell_states = next_cell_states

        return output_states

    def predict(self, x_init, h_init, c_init, beam_size):
        """ Predicts the output sequence and returns a list of predictions 
        via beam search
        @param x_init: Initial inputs
        @param h_init: Initial hidden values
        @param c_init: Initial cell states
        @param beam_size: Beam size to search through, size 1 means greedy search
        """

        # lets define a helper function that does a single LSTM tick
        def LSTMtick(x, h_curr, c_curr):
            """ Does a single tick of the LSTM network 
            @param x: single input (index in word embedding matrix)
            @param h_prev: hidden activations from previous tick
            @param c_prev: cell states form previous tick
            """

              # TODO: Initial x value should be the special <START> symbol in embedding matrix
            x_t = x_init
            h_prev = h_curr[0]
            c_prev = c_curr[0]

            hidden_states = []
            cell_states = []
            # Feed forward the neural network for input layer
           
            input_t = EmbedUnit.forward(self.embed_model, x_t)

            # Get the new hidden and cell states
            hidden_state, cell_state = LSTMUnit.forward(self.input_model, input_t, h_prev, c_prev)

            # Add them to array (to be used in the next layer)
            hidden_states.append(hidden_state)
            cell_states.append(cell_state)

            # Now continue for the hidden layers
            for i in range(1,  self.num_layers):
                input_size = len(hidden_states)
                h_prev = h_init[i]
                c_prev = c_init[i]

                # Feed forward the neural network for input layer
                x_t = hidden_state
                    
                # Get the new hidden and cell states
                hidden_state, cell_state = LSTMUnit.forward(self.hidden_model, x_t, h_prev, c_prev)

                # Append them to the list
                hidden_states.append(hidden_state)
                cell_states.append(cell_state)

                # If this is the final layer get the softmax activation
                if i == self.num_layers - 1:
                    output_state = SoftmaxUnit.forward(self.output_model, hidden_state)

          
            return (output_state, hidden_states, cell_states) # return output, new hidden, new cell

        # Copied from Karpathy's code: see utils/lstm_model.py
        # perform BEAM search. NOTE: I am not very confident in this implementation since I don't have
        # a lot of experience with these models. This implements my current understanding but I'm not
        # sure how to handle beams that predict END tokens. TODO: research this more.
        h = h_init
        c = c_init
        if beam_size > 1:
          # log probability, indices of words predicted in this beam so far, and the hidden and cell states
          beams = [(0.0, [], h, c)] 
          nsteps = 0
          while True:
            beam_candidates = []
            for b in beams:
              ixprev = b[1][-1] if b[1] else 0 # start off with the word where this beam left off
              if ixprev == 0 and b[1]:
                # this beam predicted end token. Keep in the candidates but don't expand it out any more
                beam_candidates.append(b)
                continue
              (y1, h1, c1) = LSTMtick(ixprev, b[2], b[3])
              y1 = y1.ravel() # make into 1D vector
              maxy1 = np.amax(y1)
              e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
              p1 = e1 / np.sum(e1)
              y1 = np.log(1e-20 + p1) # and back to log domain
              top_indices = np.argsort(-y1)  # we do -y because we want decreasing order
              for i in xrange(beam_size):
                wordix = top_indices[i]
                beam_candidates.append((b[0] + y1[wordix], b[1] + [wordix], h1, c1))
            beam_candidates.sort(reverse = True) # decreasing order
            beams = beam_candidates[:beam_size] # truncate to get new beams
            nsteps += 1
            if nsteps >= 20: # bad things are probably happening, break out
              break
          # strip the intermediates
          predictions = [(b[0], b[1]) for b in beams]
        else:
          # greedy inference. lets write it up independently, should be bit faster and simpler
          ixprev = x_init
          nsteps = 0
          predix = []
          predlogprob = 0.0
          while True:
            (y1, h, c) = LSTMtick(ixprev, h, c)
            ixprev, ixlogprob = self.ymax(y1)
            predix.append(ixprev)
            predlogprob += ixlogprob
            nsteps += 1
            if ixprev == 0 or nsteps >= 20:
              break
          predictions = [(predlogprob, predix)]

        return predictions


    def ymax(self, y):
         """ simple helper function here that takes unnormalized logprobs """
         y1 = y.ravel() # make sure 1d
         maxy1 = np.amax(y1)
         e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
         p1 = e1 / np.sum(e1)
         y1 = np.log(1e-20 + p1) # guard against zero probabilities just in case
         ix = np.argmax(y1)
         return (ix, y1[ix])
