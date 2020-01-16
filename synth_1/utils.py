import mxnet as mx
from mxnet import nd
import numpy as np
import json


class data_iterator (object):

    def __init__ (self, json_file, batch_size, enc_num_down, file_rate_info, num_phonemes, if_sort = True, sort_rule = 'input'):

        self.json_file = json_file
        self.batch_size = batch_size
        self.enc_num_down = enc_num_down
        self.file_rate_info = file_rate_info    
        self.num_phonemes = num_phonemes
        #self.num_tones = num_tones
        #self.num_segs = num_segs

        f = open (self.json_file, 'r')
        self.data = json.load (f)
        f.close ()

        self.keys = self.data.keys ()

        if if_sort:

            if sort_rule == 'input':

                self.keys = sorted (self.keys, key = lambda x : self.data[x]['length_input'], reverse = True)

            else:

                print 'sort_rule not existed'
                exit ()

        rate_info = np.load (file_rate_info).item () 

        self.maximal_rate = rate_info['max_rate']

    def reset (self):

        self.global_index = 0
        self.end_loading = False


    def load_one_batch (self):

        phoneme_tokens = []
        segs = []
        #tone_tokens = []

        #orders = []
        ratios = []
        for i in range (self.global_index, self.global_index + self.batch_size):

            phoneme_token = np.array ([int (item) for item in self.data[self.keys[i]]['token_phoneme'].split (' ')]).astype (np.float32)  

            ratio = self.data[self.keys[i]]['length_output'] * 1.0 / self.data[self.keys[i]]['length_input']

            seg =  self.data[self.keys[i]]['seg'].split (' ')
            seg = np.expand_dims (np.array ([float (item) for item in seg]).astype (np.float32), axis = 1)
            reverse_seg =  self.data[self.keys[i]]['reverse_seg'].split (' ')
            reverse_seg = np.expand_dims (np.array ([float (item) for item in reverse_seg]).astype (np.float32), axis = 1)
            ratios.append (ratio)
            segs.append (np.concatenate ((seg, reverse_seg), axis = 1))

            phoneme_tokens.append (phoneme_token)
 
            #tone_tokens.append (tone_token)

            #orders.append (order)

        enc_masks = [np.ones_like (item) for item in phoneme_tokens]

        max_len_input = max ([item.shape[0] for item in phoneme_tokens]) 

        input_len_needed = np.power (2, self.enc_num_down) # Since the input length is restricted to be smaller than this value     
        print phoneme_tokens
        phoneme_tokens = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (self.num_phonemes)) for item in phoneme_tokens] 
        print phoneme_tokens
        #tone_tokens = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (self.num_tones)) for item in tone_tokens]

        #orders = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (self.num_orders)) for item in orders]
        segs = [np.pad (item, ((0, input_len_needed - item.shape[0]), (0, 0)), 'constant', constant_values = (0)) for item in segs]
        enc_masks = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (0)) for item in enc_masks]

        self.global_index += self.batch_size

        if self.global_index + self.batch_size > len (self.keys):

            self.end_loading = True 

        return np.array (phoneme_tokens), np.array (segs), np.array (enc_masks),np.array (ratios)

        
def make_values_L (range_min, range_max, L, batch_size):

    logs_L = np.linspace (0, np.log (range_max * 1.0 / range_min), num = L / 2)

    values_L = nd.array (1.0 / range_min * np.exp (-logs_L))

    values_L = nd.expand_dims (nd.expand_dims (values_L, axis = 0), axis = 2)

    return nd.broadcast_axis (values_L, axis = 0, size = batch_size)


def make_dynamic_dec (T, values_L):

    values_T = nd.array (np.linspace (1, T, num = T), ctx = values_L.context)

    values_T = nd.expand_dims (nd.expand_dims (values_T, axis = 0), axis = 2)

    values_T = nd.broadcast_axis (values_T, axis = 0, size = values_L.shape[0])

    values_TL = nd.batch_dot (values_T, values_L, transpose_b = True)

    values_sin = nd.sin (values_TL)
    values_cos = nd.cos (values_TL)

    return nd.concat (values_sin, values_cos, dim = 2)

 
