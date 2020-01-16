import mxnet as mx
from mxnet import nd, gluon, gpu, cpu, autograd
import numpy as np
from modules import *
from utils_test_train import *










def AttentionWidth (rate):

    left = nd.concat (rate[:, : 1, :], rate[:, : -1, :], dim = 1)
    right = nd.concat (rate[:, 1 :, :], rate[:, -1 :, :], dim = 1)
    return 0.5 * rate + 0.25 * (left + right)
















































if __name__ == '__main__':

    batch_size = 1

    ctx = cpu (0)

    json_file = './test_train.json'

    enc_num_down = 6

    file_rate_info = './N20_rate_info.npy'

    num_phonemes = 95

    num_tones = 7

    num_orders = 2

    aco_shape = 193

    dim_embed_phoneme = 512

    dim_embed_tone = 128

    dim_embed_order = 32

    size_enc = 512

    size_dec = 512

    dp_dec = 0.15

    rate_info = np.load (file_rate_info).item ()    

    data_iter = data_iterator (json_file, batch_size, enc_num_down, file_rate_info, num_phonemes, num_tones, num_orders)    

    data_iter.reset ()

    model = End2End_1 (num_phonemes, num_tones, num_orders, enc_num_down, dim_embed_phoneme, dim_embed_tone, dim_embed_order, size_enc, size_dec, aco_shape, dp_dec, 'model_' )

    model.collect_params ().load ('parameters/model_1255',ctx=ctx)

    values_L = make_values_L (1.0, 10000.0, 512, batch_size)

    while not data_iter.end_loading:

        P, T, O, EM, R, ids = data_iter.load_one_batch ()

        T_enc = P.shape[1]

        dy_dec = make_dynamic_dec (int (T_enc * rate_info['max_rate']), values_L) 

        P = nd.array (P, ctx = ctx)

        T = nd.array (T, ctx = ctx)

        O = nd.array (O, ctx = ctx)

        EM = nd.array (EM, ctx = ctx)

        dy_dec.as_in_context (ctx)

        values_L.as_in_context (ctx)

        ratio = nd.array (R, ctx = ctx)

        pred, rate_x = model (P, T, O, T_enc, dy_dec, values_L, EM, ratio) 
        print AttentionWidth (rate_x[:, : 29, :]), ids[0], T
        #exit ()
        pred.asnumpy ().reshape ((-1, 193)).astype (np.float32).tofile (ids[0] + '.cmp')
       
          









