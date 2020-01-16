import mxnet as mx
from mxnet import nd, gluon, gpu, cpu, autograd
import numpy as np
from modules import *
from utils import *
import os
import sys







def synth_main(model_name):
    
    batch_size = 1

    ctx = cpu (0)

    json_file = './valid_data_initial.json'

    enc_num_down = 4

    file_rate_info = './data_rate_info.npy'

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
    model.collect_params ().load ('parameters/'+model_name,ctx=ctx)

    values_L = make_values_L (1.0, 10000.0, 512, batch_size)

    index = -1

    while not data_iter.end_loading:

        index += 1

        P, S, EM, R  = data_iter.load_one_batch ()

        T_enc = P.shape[1]

        dy_dec = make_dynamic_dec (int (T_enc * rate_info['max_rate']), values_L) 

        P = nd.array (P, ctx = ctx)

        S = nd.array (S, ctx = ctx)

        #O = nd.array (O, ctx = ctx)

        EM = nd.array (EM, ctx = ctx)

        dy_dec.as_in_context (ctx)

        values_L.as_in_context (ctx)

        #ratio = nd.array ([rate_info['median_rate']], ctx = ctx)
        ratio = nd.array (R, ctx = ctx)

        pred, _ = model (P, S,  T_enc, dy_dec, values_L, EM, ratio) 
        pred.asnumpy ().reshape ((-1, 187)).astype (np.float32).tofile (model_name+"/"+str (index) + '.cmp')















if __name__ == '__main__':
    models_name=sys.argv[1]
    models_name=models_name.split(',')

    for model_name in models_name:
        if os.path.exists(model_name):
            os.system("rm -r "+model_name)
            os.system("mkdir "+model_name)
        else:
            os.system("mkdir "+model_name)
        synth_main(model_name)


       
          









