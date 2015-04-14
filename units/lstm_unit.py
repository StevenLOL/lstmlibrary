from utils import functions
import numpy as np

class LSTMUnit:
   """ A single unit in a LSTM encoder layer """
   @staticmethod
   def init(input_size, hidden_size):
        model = {}

        Wxi = functions.initw(input_size, hidden_size)
        Whi = functions.initw(hidden_size, hidden_size)
        bi =  functions.initb(hidden_size)

        Wxf = functions.initw(input_size, hidden_size)
        Whf = functions.initw(hidden_size, hidden_size)
        bf = functions.initb(hidden_size)

        Wxo = functions.initw(input_size, hidden_size)
        Who = functions.initw(hidden_size, hidden_size)
        bo = functions.initb(hidden_size)

        Wxc = functions.initw(input_size, hidden_size)
        Whc = functions.initw(hidden_size, hidden_size)
        bc = functions.initb(hidden_size)

        model['Wxi'] = Wxi
        model['Whi'] = Whi
        model['bi'] = bi

        model['Wxf'] = Wxf
        model['Whf'] = Whf
        model['bf'] = bf

        model['Wxo'] = Wxo
        model['Who'] = Who
        model['bo'] = bo

        model['Wxc'] = Wxc
        model['Whc'] = Whc
        model['bc'] = bc

        return model

   @staticmethod 
   def backward(model, dc_t, dh_t, h_t, c_t, g_t, o_t, f_t, i_t, t):
       """ Computes a backward pass of the lstm network and accumulates gradients 
       @param model: Contains all the parameters of the LSTM
       @param loss_ct: The gradient of loss with respect to c_t[t]
       @param loss_ht: The gradient of loss with respect to h_t[t]
       @param h_t: Array of hidden activations
       @param c_t: Array of cell activations
       @param g_t: Array of g activations
       @param o_t: Array of o activations
       @param f_t: Array of f activations
       @param i_t: Array of i activations
       @param t: Time step to compute backward activation on
       """

       # unpack model 
       Wxi = model['Wxi']
       Whi = model['Whi']
       bi = model['bi']

       Wxf = model['Wxf']
       Whf = model['Whf']
       bf = model['bf']

       Wxo = model['Wxo']
       Who = model['Who']
       bo = model['bo']

       Wxc = model['Wxc']
       Whc = model['Whc']
       bc = model['bc']

       # Recompute tanhc_t
       tanh_c_t = np.tanh(c_t[t])

       loss_ot = np.multiply(dh_t[t], tanh_c_t)
       loss_ct = np.multiply(dh_t[t], np.multiply(o_t[t], 1-loss_ht**2))
       dc_t[t-1] = np.multiply(dc_t[t], f_t[t])
       loss_gt = np.multiply(loss_ct, i_t[t])
       loss_ft = np.multiply(loss_ct, c_t[t-1])
       loss_it = np.multiply(loss_ct, g_t[t]) 

       delta_it = loss_it * np.multiply(i_t[t], 1-i_t[t])
       dWxi = delta_it * x_t[t]
       dWhi = delta_it * h_t[t-1]
       dbi = delta_it * x_t[t]
       dh_t[t-1] += Whi * delta_it
       dx_t[t-1] += Wxi * delta_it

       delta_ft = loss_ft * np.multiply(f_t[t], 1-f_t[t])
       dWxi = delta_ft * x_t[t]
       dWhi = delta_ft * h_t[t-1]
       dbi = delta_ft * x_t[t]
       dh_t[t-1] += Whf * delta_ft
       dx_t[t-1] += Wxf * delta_ft

       delta_ot = loss_ot * np.multiply(o_t[t], 1-o_t[t])
       dWxi = delta_ot * x_t[t]
       dWhi = delta_ot * h_t[t-1]
       dbi = delta_ot * x_t[t]
       dh_t[t-1] += Who * delta_ot
       dx_t[t-1] += Wxo * delta_ot

       delta_gt = loss_gt * np.multiply(g_t[t], 1-g_t[t])
       dWxi = delta_gt * x_t[t]
       dWhi = delta_gt * h_t[t-1]
       dbi = delta_gt * x_t[t]
       dh_t[t-1] += Whi * delta_gt
       dx_t[t-1] += Wxi * delta_gt

   @staticmethod
   def forward(model, x_t, h_prev, c_prev, backward = False):
       # Return all activations if will be computing backward pass afterwards

       # unpack model 
       Wxi = model['Wxi']
       Whi = model['Whi']
       bi = model['bi']

       Wxf = model['Wxf']
       Whf = model['Whf']
       bf = model['bf']

       Wxo = model['Wxo']
       Who = model['Who']
       bo = model['bo']

       Wxc = model['Wxc']
       Whc = model['Whc']
       bc = model['bc']

       """ Computes a single forward pass using the model """
       i_t = functions.sigmoid(x_t.dot(Wxi) + h_prev.dot(Whi) + bi)
       f_t = functions.sigmoid(x_t.dot(Wxf) + h_prev.dot(Whf) + bf)
       o_t = functions.sigmoid(x_t.dot(Wxo) + h_prev.dot(Who) + bo)
       g_t = np.tanh(x_t.dot(Wxc) + h_prev.dot(Whc) + bc)
       c_t = np.multiply(f_t, c_prev) + np.multiply(i_t, g_t)
       h_t = np.multiply(o_t, c_t)

       if backward:
           return i_t, f_t, o_t, g_t, c_t, h_t
       else:
           return h_t, c_t