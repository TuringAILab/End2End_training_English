import mxnet as mx
from mxnet import nd, gluon, gpu, cpu, autograd
import numpy as np
from modules_1 import *
from utils_1 import *
import os
import sys












if __name__ == '__main__':
    batch_size = 1

    ctx = cpu (0)

    json_file = './test.json'

    enc_num_down = 6

    dec_num_down = 8

    file_rate_info = './data_rate_info.npy'

    num_phonemes = 42
    aco_path = '/home/data/LJ-speech/normed_cmp/'
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
    '''    
    with open('file_nameMap.txt','r') as f:
        file_word_map=f.readlines()
    file_word_dict={}
    for line in file_word_map:
        line = line.split('\t')
        file_word_dict[line[1].strip()]=line[0].strip()
    '''
    data_iter = data_iterator (json_file, aco_path, batch_size, enc_num_down,dec_num_down, file_rate_info, num_phonemes, aco_shape)    

    data_iter.reset ()

    model = End2End_1 (num_phonemes , enc_num_down,dec_num_down, dim_embed_seg,dim_embed_phoneme, size_enc, size_dec, aco_shape, dp_dec, 'model_' )

    model.collect_params ().load ('parameters/model_280',ctx=ctx)

    values_L = make_values_L (1.0, 10000.0, 512, batch_size)

    index = -1

    while not data_iter.end_loading:

        index += 1

        P, SEGS, EM, input_lengths  = data_iter.load_one_batch ()
        T_dec = (int (input_lengths[0] * rate_info['max_rate']) // np.power (2, dec_num_down) + 1) * np.power (2, dec_num_down) 
        DM = nd.ones (shape = (1, T_dec))

        T_enc = P.shape[1]

        dy_dec = make_dynamic_dec (T_dec, values_L)

        y_index = nd.array (np.arange (T_dec))

        y_index = nd.broadcast_axis (nd.expand_dims (y_index, axis = 0), axis = 0, size = 1)

        P = nd.array (P, ctx = ctx)

        SEGS = nd.array (SEGS, ctx = ctx)

        #O = nd.array (O, ctx = ctx)

        EM = nd.array (EM, ctx = ctx)

        dy_dec.as_in_context (ctx)

        values_L.as_in_context (values_L)

        ratio = nd.array ([rate_info['median_rate']], ctx = ctx)

        attention, output_length, atten_index,_ = model (P, SEGS, T_enc, dy_dec, values_L, EM, ratio, DM, y_index)
	
        DM[:, int (output_length.asnumpy ()) + 10 :] = 0

        rel_info = model.rel (atten_index, EM, DM, T_enc, y_index)
	
	
        DM = nd.expand_dims (DM, axis = 2)
	temp = nd.concat (attention, rel_info, dim = 2)
        pred = model.U_Net_dec (model.dense_dec (temp) * DM)

        pred.asnumpy ().reshape ((-1, 187))[: int (output_length.asnumpy ()) + 10, :].astype (np.float32).tofile (str (index) + '.cmp')
                                                     
	if index > 5:
	    break
       
          









