#coding:utf-8
import numpy as np
with open('word_and_phone','r') as f:
	lines=f.readlines()

with open('phoneme_standard.txt','r') as f2:
	phoneme_standards=f2.readlines()
phoneme_map={}
for i,phoneme_standard in enumerate(phoneme_standards):
	phoneme_map[phoneme_standard.strip()]=str(i)
words=[]
all_phonemes=[]
segs=[]
reverse_segs=[]
length_input=[]
token_phonemes=[]
for line in lines:
	line=line.split('|')
	words.append(line[0])
	seg=[]
	reverse_seg=[]
	word_phoneme=[]
	line[1]='sil '+line[1].strip()+' sil'
	if ';' in line[1]:
		phonemes=line[1].split(';')
	else:
		phonemes=[line[1]]

	for phoneme in phonemes:
		phoneme=phoneme.split(' ')
		print phoneme,'phoneme'
		for i,p in enumerate(phoneme):
			#print 1,len(phoneme),i
			if i==0 or i==len(phoneme)-1:
                                print i
				seg.append(str(0.0))
				reverse_seg.append(str(1))
			else:
				seg.append(str(round(1.0/(len(phoneme)-2)*(i-1),5)))
				reverse_seg.append(str(round(1-1.0/(len(phoneme)-2)*(i-1),5)))
		word_phoneme+=phoneme
	token_phoneme=[]
	print line[0],word_phoneme
	for word_phoneme_split in word_phoneme:
		token_phoneme.append(phoneme_map[word_phoneme_split])
	length_input.append(len(word_phoneme))
	token_phoneme=' '.join(token_phoneme)
	word_phoneme=' '.join(word_phoneme)
	seg=' '.join(seg)
	reverse_seg=' '.join(reverse_seg)
	segs.append(seg)
	reverse_segs.append(reverse_seg)
	all_phonemes.append(word_phoneme)
	token_phonemes.append(token_phoneme)
	#print words,segs,reverse_segs,length_input,all_phonemes
length_output=[]


with open('file_nameMap.txt','r') as f:
	file_word_map=f.readlines()
file_word_dict={}
for line in file_word_map:
	line = line.split('\t')
	file_word_dict[line[1].strip()]=line[0].strip()
for word in words:
	cmp_len=np.fromfile('/home/data/emo_eng/out_dio24_rmsilence/normed_cmp/'+file_word_dict[word]+'.cmp',dtype=np.float32).shape[0]/193
	length_output.append(cmp_len)
import json
data_dict={}
for i in range(len(words)):
	element_data={}
	element_data["utter_id"]=words[i]
	element_data["length_input"]=length_input[i]
	element_data["length_output"]=length_output[i]
	element_data["phoneme"]=all_phonemes[i]
	element_data["seg"]=segs[i]
	element_data["reverse_seg"]=reverse_segs[i]
	element_data["token_phoneme"]=token_phonemes[i]
	data_dict[str(i+10000)]=element_data

with open('data.json','w') as fp:
	json.dump(data_dict, fp, indent=4, sort_keys=True,ensure_ascii=False)

#with open('data.json','r') as fp:
    #print (json.load(fp))#, fp, indent=4, sort_keys=True)


