import mxnet as mx
from mxnet import gluon, nd
from U_Net import U_Net
import numpy as np


class Attention (gluon.HybridBlock):

    def __init__ (self, units, prefix):

        super (Attention, self).__init__ (prefix = prefix)

        with self.name_scope ():

            self.units = units
 
            #self.dense = gluon.nn.Dense (units, activation = 'tanh', flatten = False, prefix = 'atten_dense_')

    def hybrid_forward (self, F, key, query, embedding, mask_enc):

        atten = F.batch_dot (query, key, transpose_b = True) / np.sqrt (self.units)

        mask_enc = F.expand_dims (mask_enc, axis = 1)

        atten = F.broadcast_minus (lhs = atten, rhs = (1.0 - mask_enc) * 1e5)

        attenMax = F.max (atten, axis = 2, keepdims = True)

        index = atten >= attenMax

        return F.batch_dot (index, embedding), index

        #return self.dense (F.batch_dot (F.softmax (atten * 2, axis = 2), embedding))


class AddRNN (gluon.rnn.HybridRecurrentCell):

    def __init__ (self, prefix):

        super (AddRNN, self).__init__ (prefix = prefix)

        self._hidden_size = 1

    def state_info (self, batch_size = 0):

        return [{'shape' : (batch_size, self._hidden_size), '__layout__' : 'NC'}]

    def hybrid_forward (self, F, x, s):

        return x + s[0], [x + s[0]]


class DynamicPosEnc (gluon.HybridBlock):

    def __init__ (self, prefix):

        super (DynamicPosEnc, self).__init__ (prefix = prefix)

        with self.name_scope ():

            self.Add_RNN = AddRNN ('Add_RNN_')

    def hybrid_forward (self, F, r, values_L, T):

        self.Add_RNN.reset ()

        values_T = self.Add_RNN.unroll (T, r, merge_outputs = True)[0]

        values_T = values_T - 0.5 * r

        values_TL = F.batch_dot (values_T, values_L, transpose_b = True)

        values_sin = F.sin (values_TL)

        values_cos = F.cos (values_TL)

        return F.concat (values_sin, values_cos, dim = 2)  


class ConvBlock (gluon.HybridBlock):

    def __init__ (self, dp_rate, channels, kernel, padding, prefix, dilation = 1):

        super (ConvBlock, self).__init__ (prefix = prefix)

        self.dp_rate = dp_rate
       
        with self.name_scope ():

            self.Conv1D = gluon.nn.Conv1D (channels * 2, kernel, strides = 1, padding = padding, dilation = dilation, layout = 'NCW', use_bias = True, bias_initializer = 'zeros')  

    def hybrid_forward (self, F, x):

        if self.dp_rate > 0:

            y = F.Dropout (x)

        else:

            y = x

        y = F.swapaxes (y, 1, 2)

        y = self.Conv1D (y)

        l, r = F.split (y, num_outputs = 2, axis = 1)

        l = F.swapaxes (l, 1, 2)

        r = F.swapaxes (r, 1, 2)

        y = F.sigmoid (l) * F.tanh (r)

        return (x + y) * np.sqrt (0.5)


class Relative (gluon.HybridBlock):

    def __init__ (self, prefix):

        super (Relative, self).__init__ (prefix = prefix)

        with self.name_scope ():

            self.add_rnn = AddRNN ('addrnn_rel_')

    def hybrid_forward (self, F, index, mask_enc, mask_dec, x_T, y_index):

        self.add_rnn.reset ()

        mask_dec = F.expand_dims (mask_dec, axis = 2)

        rate = F.sum (index * mask_dec, axis = 1, keepdims = False)

        rate = F.expand_dims (rate, axis = 2)

        rateSum = self.add_rnn.unroll (x_T, rate, merge_outputs = True)[0]

        rateSum = rateSum * F.expand_dims (mask_enc, axis = 2)   

        rateSum_ex = F.swapaxes (rateSum, dim1 = 1, dim2 = 2)

        y_index_ex = F.expand_dims (y_index, axis = 2)

        minus = y_index_ex - rateSum_ex

        minus_extract = - (minus >= 0) * 1e8 + minus

        shift = F.max (minus_extract, axis = 2, keepdims = True)

        rate_per_frame = F.batch_dot (index, rate)

        fraction = (shift + rate_per_frame) / rate_per_frame

        relative_posenc_1 = F.exp(- 0.5 * F.square(1.5 * fraction - 0.75) / 0.16)
        relative_posenc_2 = F.exp(- 0.5 * F.square(0.75 * fraction - 0.75) / 0.16)
        relative_posenc_3 = F.exp(- 0.5 * F.square(0.75 * fraction) / 0.16)

        return F.concat(relative_posenc_1, relative_posenc_2, relative_posenc_3, dim=2) * mask_dec





        
        

class End2End_1 (gluon.nn.HybridBlock):

    def __init__ (self, num_phoneme, num_down_enc, num_down_dec,dim_embed_seg,dim_embed_phoneme, size_enc, size_dec, size_output, dp_dec, prefix):

        super (End2End_1, self).__init__ (prefix = prefix)

        self.num_phoneme = num_phoneme
        #self.num_tone = num_tone
        #self.num_order = num_order
        self.num_down_enc = num_down_enc
        self.dim_embed_phoneme = dim_embed_phoneme
        self.dim_embed_seg = dim_embed_seg
        #self.dim_embed_tone = dim_embed_tone
        #self.dim_embed_order = dim_embed_order
        self.num_down_dec = num_down_dec
        self.size_enc = size_enc
        self.size_dec = size_dec
        self.size_output = size_output
        self.dp_dec = dp_dec  

        with self.name_scope ():

            self.emd_phoneme = gluon.nn.Embedding (self.num_phoneme, self.dim_embed_phoneme, prefix = 'embed_phoneme_') 

            self.emd_phoneme_rate = gluon.nn.Embedding (self.num_phoneme, self.dim_embed_phoneme, prefix = 'embed_phoneme_rate_') 

            self.dense_seg = gluon.nn.Dense (self.dim_embed_seg, flatten = False, prefix = 'dense_seg_')

            self.U_Net_rate = U_Net (0, 256, 1, self.num_down_enc, [self.size_enc] * self.num_down_enc, [1] * self.num_down_enc, 'tanh', 'U_Net_rate_', dropout = 0.0)

            self.dense_enc_1 = gluon.nn.Dense (self.size_enc, activation = 'tanh', flatten = False, prefix = 'dense_enc_1')

            self.dense_enc_2 = gluon.nn.Dense (self.size_enc, activation = 'tanh', flatten = False, prefix = 'dense_enc_2')

            self.dense_enc_3 = gluon.nn.Dense (self.size_enc, activation = 'tanh', flatten = False, prefix = 'dense_enc_3')

            self.Atten = Attention (self.size_dec, 'atten_')

            self.Dynamic = DynamicPosEnc ('DynamicPosEnc_')

            self.rel = Relative ('rel_')

            self.dense_dec = gluon.nn.Dense (self.size_dec, flatten = False, prefix = 'dense_dec_') 

            self.U_Net_dec = U_Net (0, 256, self.size_output, self.num_down_dec, [self.size_dec] * self.num_down_dec, [1] * self.num_down_dec, 'tanh', 'U_Net_dec_', dropout = self.dp_dec)    


    def hybrid_forward (self, F, p, seg, T, dy_dec, values_L, mask_enc, ratio, mask_dec, y_index):      

        p_embed = self.emd_phoneme (p)
        x_embed = p_embed + self.dense_seg (seg)
        x_embed_rate = self.emd_phoneme_rate (p)
        ratio = F.expand_dims (F.expand_dims (ratio, axis = 1), axis = 2)

        rate_x = self.U_Net_rate (x_embed_rate * F.expand_dims (mask_enc, axis = 2))

        rate_x = F.maximum (F.relu (rate_x + ratio), 2.0) 

        enc = self.dense_enc_1 (x_embed * F.expand_dims (mask_enc, axis = 2)) 

        enc = self.dense_enc_2 (enc)

        enc = self.dense_enc_3 (enc)

        dy_enc = self.Dynamic (rate_x, values_L, T)

        attention, atten_index = self.Atten (dy_enc, dy_dec, enc, mask_enc) 

        rel_info = self.rel (atten_index, mask_enc, mask_dec, T, y_index)

        mask_dec = F.expand_dims (mask_dec, axis = 2)
        mask_enc = F.expand_dims (mask_enc, axis = 2)
        return attention, F.sum (rate_x * mask_enc, axis = [1, 2]), atten_index,  rate_x
        #return F.sum (rate_x * mask_enc, axis = [1, 2]),self.U_Net_dec (self.dense_dec (F.concat (attention, rel_info, dim = 2)) * mask_dec), rate_x 

        #return self.final_dense (y), rate_x



class Loss_pred (gluon.HybridBlock):

    def __init__ (self, prefix):

        super (Loss_pred, self).__init__ (prefix = prefix)

    def hybrid_forward (self, F, pred, truth, mask):

        mask_expand = F.expand_dims (mask, axis = 2)

        return F.sum (F.square (pred - truth) * mask_expand, axis = [1, 2], keepdims = False) / F.sum (mask, axis = 1, keepdims = False)    


class Loss_rate (gluon.HybridBlock):

    def __init__ (self, prefix):

        super (Loss_rate, self).__init__ (prefix = prefix)

    def hybrid_forward (self, F, pred_rate, true_rate, mask_enc):

        mask_enc = F.expand_dims (mask_enc, axis = 2)

        return F.maximum (F.abs (F.sum (pred_rate * mask_enc, axis = [1, 2]) - true_rate), 20.0)  























