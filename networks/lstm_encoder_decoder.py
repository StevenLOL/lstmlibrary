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

    def predict(self, Xi, beam_size):
        """ Generates a sequence for given input and beam size 
        @param input: Input sequence to convert
        @param beam_size: Beam size for decoding
        """
        
        h_out, c_out = self.encodingLayer.forward(Xi)
        res = self.decodingLayer.predict(0, h_out, c_out, beam_size)

        return res
        # First get vector representation
        h_out, c_out = self.encodingLayer.forward(Xi)

        # Copied from Karpathy's code: see utils/lstm_model.py
        # perform BEAM search. NOTE: I am not very confident in this implementation since I don't have
        # a lot of experience with these models. This implements my current understanding but I'm not
        # sure how to handle beams that predict END tokens. TODO: research this more.
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
              (y1, h1, c1) = LSTMtick(Ws[ixprev], b[2], b[3])
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
          ixprev = 0
          nsteps = 0
          predix = []
          predlogprob = 0.0
          while True:
            (y1, h, c) = LSTMtick(Ws[ixprev], h, c)
            ixprev, ixlogprob = ymax(y1)
            predix.append(ixprev)
            predlogprob += ixlogprob
            nsteps += 1
            if ixprev == 0 or nsteps >= 20:
              break
          predictions = [(predlogprob, predix)]

        return predictions



