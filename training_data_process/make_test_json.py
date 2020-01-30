#coding:utf-8
import numpy as np
with open('testword_and_phone','r') as f:
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
		for i,p in enumerate(phoneme):
			#print 1,len(phoneme),i
			seg.append(str(round(1.0/len(phoneme)*i,5)))
			reverse_seg.append(str(round(1-1.0/len(phoneme)*(i),5)))
		word_phoneme+=phoneme
	token_phoneme=[]
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
import json
data_dict={}
for i in range(len(words)):
	element_data={}
	element_data["utter_id"]=words[i]
	element_data["length_input"]=length_input[i]
	element_data["phoneme"]=all_phonemes[i]
	element_data["seg"]=segs[i]
	element_data["reverse_seg"]=reverse_segs[i]
	element_data["token_phoneme"]=token_phonemes[i]
	data_dict[str(i+10000)]=element_data

with open('testdata.json','w') as fp:
	json.dump(data_dict, fp, indent=4, sort_keys=True,ensure_ascii=False)

with open('testdata.json','r') as fp:
    print (json.load(fp))#, fp, indent=4, sort_keys=True)


