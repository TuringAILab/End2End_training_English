#coding:utf-8
import numpy as np
import re
from engFrontEnd.frontEndTool import frontEndTool

class Gen_data(object):
	def __init__(self):
		self.frontTool = frontEndTool ()

		with open('phoneme_standard.txt','r') as f2:
			phoneme_standards=f2.readlines()
		self.phoneme_map={}
		for i,phoneme_standard in enumerate(phoneme_standards):
			self.phoneme_map[phoneme_standard.strip()]=str(i)
		'''
		with open('file_nameMap.txt','r') as f:
			file_word_map=f.readlines()
		self.file_word_dict={}
		for line in file_word_map:
			line = line.split('\t')
			self.file_word_dict[line[1].strip()]=line[0].strip()
		'''
	def gen_phoneme_and_order(self, line):


		#line = line.split ('\n')[0]
		index, line = line.strip().split('\t')
		lineList = self.frontTool.preprocessText(line)
		phonemeLines = self.frontTool.getPhoneListwithPau(lineList)
		dict_mgc_lf0_bap_list = []
		modelTaskList = []
		for phonemeLine in phonemeLines:
			if not re.match(r'pau\d',phonemeLine):
				phoneme, order,lenSingle = self.frontTool.processPhonemeLine(phonemeLine)
				#print phoneme, order,lenSingle,phonemeLine,'phonemeLine'
				modelTaskList.append((phoneme, order,lenSingle))
			else:
				modelTaskList.append(phonemeLine) # add "pau$"
		concatTaskList=[]
		phonemes=['sil']
		orders=[str(0)]
		token_phonemes=[str(40)]
		for task in modelTaskList:
			if type(task).__name__=="str":
				#paramsList.append(task)
				continue
			phoneme,order,len_sentence=task
			for v in phoneme:
				if v=='.':
					phonemes.append('sp')
					token_phonemes.append(str(41))
				else:
					phonemes.append(v)
					token_phonemes.append(str(self.phoneme_map[v.lower()]))
			for v in order:
				if v=='.':
					orders.append(str(0))
				else:
					orders.append(str(v))
		if phonemes[-1]=='sp':
			phonemes[-1]='sil'
			token_phonemes[-1]="40"
		try:		
		    cmp_len=np.fromfile('/home/data/emo_eng/16k_sent/out/normed_cmp/'+index+'.cmp',dtype=np.float32).shape[0]/187
		    length_output=cmp_len
        	except:
		    return 0
		#length_output = 0
		reverse_seg = [str(1 - float(i)) for i in orders]
		return ' '.join(phonemes),' '.join(orders),' '.join(reverse_seg),' '.join(token_phonemes),len(token_phonemes),length_output, index

gen_data=Gen_data()





import json


with open('emo_sent.txt','r') as f:
	lines=f.readlines()

data_dict={}
for i,line in enumerate(lines):
	line=line.strip()
        try:	
	    phoneme,seg,reverse_seg,token_phoneme,length_input,length_output, index=gen_data.gen_phoneme_and_order(line)
	
	except:
            #print(line.split('\t')[0])
	    continue
		
	element_data={}
	element_data["utter_id"]=index
	element_data["length_input"]=length_input
	element_data["length_output"]=length_output
	element_data["phoneme"]=phoneme
	element_data["seg"]=seg
	element_data["reverse_seg"]=reverse_seg
	element_data["token_phoneme"]=token_phoneme
	data_dict[str(i+10000)]=element_data

with open('emo_sent_data.json','w') as fp:
	json.dump(data_dict, fp, indent=4, sort_keys=True,ensure_ascii=False)

#with open('data.json','r') as fp:
    #print (json.load(fp))#, fp, indent=4, sort_keys=True)


