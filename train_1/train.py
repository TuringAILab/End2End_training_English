import mxnet as mx
from mxnet import nd, gluon, gpu, cpu, autograd
import numpy as np
from modules import *
from utils import *



def train (model, loss_pred, loss_rate, trainer, batch_size_train, batch_size_valid, data_iter_train, data_iter_valid, ctx, maximal_epoch, epoch_save, path_save):

    curr_epoch = 0

    f = open ('record', 'a+')

    loss_rate_weight = 0.01 
    
    values_L_train = make_values_L(1.0, 10000.0, 512, batch_size_train)
    values_L_train = gluon.utils.split_and_load (values_L_train, ctx)

    values_L_valid = make_values_L(1.0, 10000.0, 512, batch_size_valid)
    values_L_valid = gluon.utils.split_and_load (values_L_valid, ctx)

    while curr_epoch < maximal_epoch:

        train_epoch_loss_pred = []
        train_epoch_loss_rate = []

        data_iter_train.reset ()    

        #values_L = make_values_L (1.0, 10000.0, 512, batch_size_train)

        while not data_iter_train.end_loading:

            phonemes, segs, enc_masks, acos, dec_masks, aco_lengths, ratios = data_iter_train.load_one_batch ()  #enc_masks=phoneme_token

            T_enc = phonemes.shape[1]

            #dy_dec = make_dynamic_dec (acos.shape[1], values_L)
  
            phonemes = gluon.utils.split_and_load (phonemes, ctx)
            #tones = gluon.utils.split_and_load (tones, ctx)
            segs = gluon.utils.split_and_load (segs, ctx)
            #reverse_segs = gluon.utils.split_and_load (reverse_segs, ctx)
            enc_masks = gluon.utils.split_and_load (enc_masks, ctx)
            acos = gluon.utils.split_and_load (acos, ctx)
            dec_masks = gluon.utils.split_and_load (dec_masks, ctx)
            #values_L_list = gluon.utils.split_and_load (values_L, ctx)
            #dy_dec = gluon.utils.split_and_load (dy_dec, ctx)             
            aco_lengths = gluon.utils.split_and_load (aco_lengths, ctx)
            ratios = gluon.utils.split_and_load (ratios, ctx)
 
            losses = []
            losses_aco = []
            losses_rate = []

            with autograd.record ():

                for P, S, EM, A, DM, VL, AL, R in zip (phonemes, segs, enc_masks, acos, dec_masks, values_L_train, aco_lengths, ratios):

		    DD = make_dynamic_dec(A.shape[1], VL)
                    pred, pred_rate = model (P, S, T_enc, DD, VL, EM, R, DM)

                    loss = loss_pred (pred, A, DM)
                    loss_r = loss_rate (pred_rate, AL, EM) * loss_rate_weight

                    losses.append (loss + loss_r)
                    losses_aco.append (loss)
                    losses_rate.append (loss_r)

            for item in losses:

                item.backward ()

            trainer.step (batch_size_train)
            nd.waitall ()

            loss_avg_aco = 0
            loss_avg_rate = 0

            for item in losses_aco:
                loss_avg_aco += np.sum (item.asnumpy ()) / batch_size_train
            for item in losses_rate:
                loss_avg_rate += np.sum (item.asnumpy ()) / batch_size_train
            #print loss_avg_aco, loss_avg_rate
            train_epoch_loss_pred.append (loss_avg_aco)
            train_epoch_loss_rate.append (loss_avg_rate)

        if curr_epoch % epoch_save == 0:
            model.collect_params ().save (path_save + 'model_' + str (curr_epoch))


        valid_epoch_loss_pred = []
        valid_epoch_loss_rate = []

        data_iter_valid.reset ()    

        #values_L = make_values_L (1.0, 10000.0, 512, batch_size_valid)

        while not data_iter_valid.end_loading:

            phonemes, segs, enc_masks, acos, dec_masks, aco_lengths, ratios = data_iter_valid.load_one_batch ()  

            T_enc = phonemes.shape[1]

            #dy_dec = make_dynamic_dec (acos.shape[1], values_L)
  
            phonemes = gluon.utils.split_and_load (phonemes, ctx)
            #tones = gluon.utils.split_and_load (tones, ctx)
            segs = gluon.utils.split_and_load (segs, ctx)
            #reverse_segs = gluon.utils.split_and_load (reverse_segs, ctx)
            enc_masks = gluon.utils.split_and_load (enc_masks, ctx)
            acos = gluon.utils.split_and_load (acos, ctx)
            dec_masks = gluon.utils.split_and_load (dec_masks, ctx)
            #values_L_list = gluon.utils.split_and_load (values_L, ctx)
            #dy_dec = gluon.utils.split_and_load (dy_dec, ctx)             
            aco_lengths = gluon.utils.split_and_load (aco_lengths, ctx)
            ratios = gluon.utils.split_and_load (ratios, ctx)
 
            losses = []
            losses_aco = []

            losses_rate = []

            for P, S, EM, A, DM, VL, AL, R in zip (phonemes, segs, enc_masks, acos, dec_masks, values_L_valid, aco_lengths, ratios):

		DD = make_dynamic_dec (A.shape[1], VL)
                pred, pred_rate = model (P, S, T_enc, DD, VL, EM, R, DM)

                loss = loss_pred (pred, A, DM)
                loss_r = loss_rate (pred_rate, AL, EM) * loss_rate_weight

                losses.append (loss + loss_r)
                losses_aco.append (loss)
                losses_rate.append (loss_r)


            loss_avg_aco = 0
            loss_avg_rate = 0

            for item in losses_aco:
                loss_avg_aco += np.sum (item.asnumpy ()) / batch_size_valid
    
            for item in losses_rate:
                loss_avg_rate += np.sum (item.asnumpy ()) / batch_size_valid

            valid_epoch_loss_pred.append (loss_avg_aco)
            valid_epoch_loss_rate.append (loss_avg_rate)

        print 'epoch :', str (curr_epoch), 'train epoch loss aco:', str (np.mean (np.array (train_epoch_loss_pred))), 'valid epoch loss aco :', str (np.mean (np.array (valid_epoch_loss_pred))), 'train epoch loss rate: ', str (np.mean (np.array (train_epoch_loss_rate))), 'valid epoch loss rate:', str (np.mean (np.array (valid_epoch_loss_rate)))        

        info = 'epoch :' + str (curr_epoch) + 'train epoch loss aco:' + str (np.mean (np.array (train_epoch_loss_pred))), 'valid epoch loss aco:', str (np.mean (np.array (valid_epoch_loss_pred))) + 'train epoch loss rate: ' + str (np.mean (np.array (train_epoch_loss_rate))) + 'valid epoch loss rate: ', str (np.mean (np.array (valid_epoch_loss_rate))) + '\n'

        f.writelines (info)

        curr_epoch += 1

    f.close ()

     
###########################################

if __name__ == '__main__':

    maximal_epoch = 5000

    epoch_save = 10

    path_save = 'parameters/'

    batch_size_train = 96

    batch_size_valid = 4

    ctx = [gpu(i) for i in range (0, 4)]

    train_json_file = './train_selected_data.json' 

    valid_json_file = './valid_selected_data.json'

    aco_path = '/home/data/emo_eng/out_dio24_rmsilence/normed_cmp/' 

    enc_num_down = 4

    file_rate_info = './word_rate_info.npy'

    num_phonemes = 42

    #num_tones = 7 

    #num_segs = 2

    aco_shape = 193

    dim_embed_phoneme = 512

    #dim_embed_tone = 128

    #dim_embed_order = 32  

    size_enc = 512

    size_dec = 512

    dp_dec = 0.5
 
    dim_embed_seg=512
    
    
    data_iter_train = data_iterator (train_json_file, aco_path, batch_size_train, enc_num_down, file_rate_info, num_phonemes, aco_shape) 
    data_iter_valid = data_iterator (valid_json_file, aco_path, batch_size_valid, enc_num_down, file_rate_info, num_phonemes, aco_shape)

    
    model = End2End_1 (num_phonemes , enc_num_down, dim_embed_seg,dim_embed_phoneme, size_enc, size_dec, aco_shape, dp_dec, 'model_' )

    loss_pred = Loss_pred ('loss_pred_')

    loss_rate = Loss_rate ('loss_rate_') 

    model.collect_params ().initialize (ctx = ctx)
    #model.collect_params ().load ('parameters/model_2900', ctx = ctx, allow_missing = True, ignore_extra = True)
    trainer = gluon.Trainer (model.collect_params (), 'adam', {'learning_rate' : 1e-4, 'clip_gradient' : 1.0})

    train (model, loss_pred, loss_rate, trainer, batch_size_train, batch_size_valid, data_iter_train, data_iter_valid, ctx, maximal_epoch, epoch_save, path_save)

    


       

