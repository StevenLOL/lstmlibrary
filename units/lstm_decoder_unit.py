from utils import functions
import numpy as np

class LSTMDecoderUnit:
   """ A single unit in a LSTM decoder layer """
   @staticmethod
   def init(input_size, hidden_size, output_size):
        """ Creates a LSTMDecoderUnit that can be used in a language model 
            @param input_size: Number of inputs in a single unit
            @param hidden_size: Size of the memory state
            @param output_size: Number of outputs of a single units
        """
        model = {}

        Wxi = functions.initw(input_size, hidden_size)
        Whi = functions.initw(hidden_size, hidden_size)
        Wxf = functions.initw(input_size, hidden_size)
        Whf = functions.initw(hidden_size, hidden_size)
        Wxo = functions.initw(input_size, hidden_size)
        Who = functions.initw(hidden_size, hidden_size)
        Wxc = functions.initw(input_size, hidden_size)
        Whc = functions.initw(hidden_size, hidden_size)

        model['Wxi'] = Wxi
        model['Whi'] = Whi
        model['Wxf'] = Wxf
        model['Whf'] = Whf
        model['Wxo'] = Wxo
        model['Who'] = Who
        model['Wxc'] = Wxc
        model['Whc'] = Whc

        return model

   @staticmethod
   def forward(model, x_t, h_prev, c_prev):
       # unpack model 
       Wxi = model['Wxi']
       Whi = model['Whi']
       Wxf = model['Wxf']
       Whf = model['Whf']
       Wxo = model['Wxo']
       Who = model['Who']
       Wxc = model['Wxc']
       Whc = model['Whc']

       """ Computes a single forward pass using the model """
       i_t = functions.sigmoid(x_t.dot(Wxi) + h_prev.dot(Whi))
       f_t = functions.sigmoid(x_t.dot(Wxf) + h_prev.dot(Whf))
       o_t = functions.sigmoid(x_t.dot(Wxo) + h_prev.dot(Who))
       g_t = np.tanh(x_t.dot(Wxc) + h_prev.dot(Whc))
       c_t = np.multiply(f_t, c_prev) + np.multiply(i_t, g_t)
       h_t = np.multiply(o_t, c_t)
       return h_t, c_t