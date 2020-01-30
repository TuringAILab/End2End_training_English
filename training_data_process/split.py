import json
import numpy as np
import random
#files=['emo_pinyin_seg']
files = ['3_emo_sent_data']
for _file in files:
	json_to_be_split = 'selected_'+_file+'.json'

	valid_num = 1

	f = open (json_to_be_split, 'r')

	data = json.load (f)

	f.close ()

	keys = data.keys ()
        valid_num = 8 #int(0.25 * len(keys))
	random.shuffle (keys)

	valid_keys = keys[- valid_num :]
	train_keys = keys[: - valid_num]


	train_json = 'train_' + json_to_be_split
	valid_json = 'valid_' + json_to_be_split

	train_dict = {}
	valid_dict = {}

	for key in train_keys:

		train_dict.update ({key : data[key]})
	
	for key in valid_keys:

		valid_dict.update ({key : data[key]})

	f = open (train_json, 'w')

	json.dump (train_dict, f, indent = 4)

	f.close ()

	f = open (valid_json, 'w')

	json.dump (valid_dict, f, indent = 4)

	f.close ()






