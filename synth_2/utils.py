import mxnet as mx
from mxnet import nd
import numpy as np
import json
import random

class data_iterator (object):

    def __init__ (self, file_word_dict,json_file, batch_size, enc_num_down, dec_num_down, file_rate_info, num_phonemes, if_sort = True, sort_rule = 'input'):
        self.file_word_dict=file_word_dict
        self.json_file = json_file
        self.batch_size = batch_size
        self.enc_num_down = enc_num_down
        self.dec_num_down = dec_num_down
        self.file_rate_info = file_rate_info    
        self.num_phonemes = num_phonemes
        #self.num_tones = num_tones
        #self.num_segs = num_segs
        self.encoderLenNeed = np.power (2, enc_num_down)
        f = open (self.json_file, 'r')
        self.data = json.load (f)
        f.close ()

        self.keys = self.data.keys ()

        if if_sort:

            if sort_rule == 'input':

                self.keys = sorted (self.keys, key = lambda x : self.data[x]['length_input'], reverse = True)

            elif sort_rule == 'output':

                self.keys = sorted (self.keys, key = lambda x : self.data[x]['length_output'], reverse = True)

            else:

                print 'sort_rule not existed'
                exit ()

        rate_info = np.load (file_rate_info).item () 

        self.maximal_rate = rate_info['max_rate']
        print self.maximal_rate
    def reset (self):

        self.global_index = 0
        self.end_loading = False
        random.shuffle (self.keys)

    def load_one_batch (self):

        token_phonemes = []
       
        #tone_tokens = []
        input_lengths = []
        segs = []

        #reverse_segs = []

        for i in range (self.global_index, self.global_index + self.batch_size):

            token_phoneme = np.array ([int (item) for item in self.data[self.keys[i]]['token_phoneme'].split (' ')]).astype (np.float32)  
            token = self.data[self.keys[i]]['token_phoneme'].split (' ')
            token = np.array ([int (item) for item in token]).astype (np.float32)
            #print token,'token'
            #tone_token = np.array ([int (item) for item in self.data[self.keys[i]]['tone'].split (' ')]).astype (np.float32)
            #seg = np.array ([item.encode('utf-8') for item in self.data[self.keys[i]]['seg'].split (' ')]).astype (np.float32)
            #reverse_seg = np.array ([item for item in self.data[self.keys[i]]['reverse_seg'].split (' ')]).astype (np.float32)

            seg =  self.data[self.keys[i]]['seg'].split (' ')
            seg = np.expand_dims (np.array ([float (item) for item in seg]).astype (np.float32), axis = 1)
            reverse_seg =  self.data[self.keys[i]]['reverse_seg'].split (' ')
            reverse_seg = np.expand_dims (np.array ([float (item) for item in reverse_seg]).astype (np.float32), axis = 1)
            #print seg,'seg'
            #print order,'order'
            segs.append (np.concatenate ((seg, reverse_seg), axis = 1))
            token_phonemes.append (token_phoneme)
            input_lengths.append (self.data[self.keys[i]]['length_input'])

            #tone_tokens.append (tone_token)

            #segs.append (seg)

            #reverse_segs.append (reverse_seg)
            #print self.keys[i]
        #print token_phonemes
        enc_masks = [np.ones_like (item) for item in token_phonemes]

        max_len_input = max ([item.shape[0] for item in token_phonemes]) 

        #input_len_needed = np.power (2, self.enc_num_down) # Since the input length is restricted to be smaller than this value     
        input_len_needed = ((max_len_input // self.encoderLenNeed) + 1) * self.encoderLenNeed

        token_phonemes = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (self.num_phonemes)) for item in token_phonemes] 

        segs = [np.pad (item, ((0, input_len_needed - item.shape[0]), (0, 0)), 'constant', constant_values = (0)) for item in segs]
        #reverse_segs = [np.pad (item, ((0, input_len_needed - item.shape[0]), (0, 0)), 'constant', constant_values = (0)) for item in reverse_segs]

        #tone_tokens = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (self.num_tones)) for item in tone_tokens]

        #segs = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (self.num_segs)) for item in segs]

        enc_masks = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (0)) for item in enc_masks]

        output_len_needed = int (max_len_input * self.maximal_rate)
        output_len_needed = (output_len_needed // np.power (2, self.dec_num_down) + 1) * np.power (2, self.dec_num_down)
        print max_len_input,self.maximal_rate



        self.global_index += self.batch_size

        if self.global_index + self.batch_size > len (self.keys):

            self.end_loading = True 

        return np.array (token_phonemes),  np.array (segs), np.array (enc_masks),input_lengths

        
def make_values_L (range_min, range_max, L, batch_size):

    logs_L = np.linspace (0, np.log (range_max * 1.0 / range_min), num = L / 2)

    values_L = nd.array (1.0 / range_min * np.exp (-logs_L))

    values_L = nd.expand_dims (nd.expand_dims (values_L, axis = 0), axis = 2)

    return nd.broadcast_axis (values_L, axis = 0, size = batch_size)


def make_dynamic_dec (T, values_L):

    values_T = nd.array (np.linspace (1, T, num = T))

    values_T = nd.expand_dims (nd.expand_dims (values_T, axis = 0), axis = 2)

    values_T = nd.broadcast_axis (values_T, axis = 0, size = values_L.shape[0])

    values_TL = nd.batch_dot (values_T, values_L, transpose_b = True)

    values_sin = nd.sin (values_TL)
    values_cos = nd.cos (values_TL)

    return nd.concat (values_sin, values_cos, dim = 2)

 
