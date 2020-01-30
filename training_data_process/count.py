import json
import fnmatch
import numpy as np
import os


def select_data (file, mini, maxi, mini_rate, maxi_rate, if_make_new_json = False):
    print file
    f = open (file, 'r')
    if file=='cameal_pinyin_seg.json':
        maxi_rate=150.0
    dict = json.load (f)
    f.close ()
    print maxi
    new_dict = {}

    for key in dict.keys ():

        input_length = dict[key]['length_input']        

        output_length = dict[key]['length_output']

        rate = output_length * 1.0 / input_length

        if input_length < mini or input_length > maxi or rate < mini_rate or rate > maxi_rate:

            continue

        new_dict.update ({key : dict[key]}) 
    print len (new_dict.keys ()) , len (dict.keys ())
    data_portion = len (new_dict.keys ()) * 1.0 / len (dict.keys ())

    if if_make_new_json:

        f = open ('selected_' + file, 'w') 

        json.dump (new_dict, f, indent = 4)

        f.close ()

    mini = 1e100

    maxi = 0

    range_min_rate = 1e100

    range_max_rate = 0

    total_input_length = 0

    total_output_length = 0    

    rates = []
    for key in new_dict.keys ():
        input_length = dict[key]['length_input']
       
        output_length = dict[key]['length_output']
        print input_length,output_length


        rate = output_length * 1.0 / input_length

        mini = min (input_length, mini)

        maxi = max (input_length, maxi)

        range_min_rate = min (rate, range_min_rate)

        range_max_rate = max (rate, range_max_rate)
         
        total_input_length += input_length

        total_output_length += output_length
        print total_input_length,total_output_length

        rates.append (rate)      


    avg_rate = total_output_length * 1.0 / total_input_length

    median_rate = np.median (rates)

    if if_make_new_json:

        rate_info = {}

        rate_info.update ({'min_rate' : range_min_rate, 'max_rate' : range_max_rate, 'avg_rate' : avg_rate, 'median_rate' : median_rate})     

        np.save (file.split ('.')[0] + '_rate_info', rate_info)

    return data_portion, range_min_rate, range_max_rate, mini, maxi, avg_rate, median_rate 



if __name__ == '__main__':


    mini = 0

    maxi = 160

    mini_rate = 15.0

    maxi_rate = 65.0 #75.0

    if_make_new_json = True

    json_files = [ '3_emo_sent_data.json']
    

    for file in json_files:
        print file
        data_portion, range_min_rate, range_max_rate, range_min, range_max, avg_rate, median_rate = select_data (file, mini, maxi, mini_rate, maxi_rate, if_make_new_json = if_make_new_json) 

        print file, data_portion, range_min_rate, range_max_rate, range_min, range_max, avg_rate, median_rate, '\n'
 

