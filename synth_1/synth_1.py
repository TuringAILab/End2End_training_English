import mxnet as mx
from mxnet import nd, gluon, gpu, cpu, autograd
import numpy as np
from modules_1 import *
from utils_1 import *
import os
import sys












if __name__ == '__main__':
    batch_size = 1

    ctx = gpu (0)

    json_file = './valid_selected_emo_sent.json'

    enc_num_down = 8

    file_rate_info = './emo_sent_rate_info.npy'

    num_phonemes = 42

    #num_tones = 7

    #num_orders = 2

    aco_shape = 187

    dim_embed_phoneme = 512

    #dim_embed_tone = 128

    #dim_embed_order = 32

    size_enc = 512

    size_dec = 512

    dp_dec = 0.15
    dim_embed_seg=512

    rate_info = np.load (file_rate_info).item ()    

    data_iter = data_iterator (json_file, batch_size, enc_num_down, file_rate_info, num_phonemes)    

    data_iter.reset ()

    model = End2End_1 (num_phonemes , enc_num_down, dim_embed_seg,dim_embed_phoneme, size_enc, size_dec, aco_shape, dp_dec, 'model_' )

    model.collect_params ().load ('parameters/model_290',ctx=ctx)

    values_L = make_values_L (1.0, 10000.0, 512, batch_size)

    index = -1

    while not data_iter.end_loading:

        #index += 1

        P, S, EM,R  = data_iter.load_one_batch ()

        T_enc = P.shape[1]

        dy_dec = make_dynamic_dec (int (T_enc * rate_info['max_rate']), values_L) 

        P = nd.array (P, ctx = ctx)
	#print np.sum(EM, axis = 1)
	#if np.sum(EM, axis=1) > 11:
	#    continue
		
        S = nd.array (S, ctx = ctx)

        #O = nd.array (O, ctx = ctx)

        EM = nd.array (EM, ctx = ctx)

        dy_dec = dy_dec.as_in_context (ctx)

        values_L = values_L.as_in_context (ctx)

        ratio = nd.array (R, ctx = ctx)

        pred, _ = model (P, S,  T_enc, dy_dec, values_L, EM, ratio) 
        pred.asnumpy ().reshape ((-1, 187)).astype (np.float32).tofile (str (index) + '.cmp')
	if index > 5:
	    break
	index += 1


       
          









