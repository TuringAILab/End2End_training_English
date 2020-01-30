# -*- coding: utf-8 -*-
import ctypes
import numpy as np
import string
import os
import re

class frontEndTool (object):
	def __init__ (self):
		# prepare tool to get phoneme
		path_self = os.path.dirname(os.path.realpath(__file__))
		self.getTextPhonemeTool = ctypes.cdll.LoadLibrary(path_self+"/libEngTTS.so").get_text_phone
		self.getTextPhonemeTool.argtypes = [ctypes.c_char_p]
		self.getTextPhonemeTool.restype = ctypes.c_char_p
		# end of prepare tool to get phoneme

	def text2Phoneme(self, lines):
		# remove \n of every utterance
		utterances = [line.split ('\n')[0] for line in lines]
		# end of remove \n of every uttrance
		# get phoneme and process every phoneme
		phonemes = []
		orders = []
		lenSingles=[]
		for line in utterances:
			lineList = self.preprocessText(line)
			phonemeLines = self.getPhoneListwithPau (lineList)
			## process phoneme and get order info
			for phonemeLine in phonemeLines:
				if not re.search(r'pau\d',phonemeLine):
					try:
						phoneme, order,lenSingle = self.processPhonemeLine(phonemeLine)
					except:
						phoneme = ['.']
						order = [0.0]
						lenSingle=[]
					phonemes.append (phoneme)
					orders.append (order)
					lenSingles.append(lenSingle)
				else:
					phonemes.append (phonemeLine)
					orders.append (160)

		return phonemes, orders,lenSingles



	def processPhonemeSingle(self, phoneme):
		wordPhonemes = phoneme.split (' ')
		wordPhonemes = [i if i[-1] not in ['0', '1', '2'] else i[: -1] for i in wordPhonemes]
		wordPhonemes = [string.upper (i) for i in wordPhonemes]
		wordPhonemes = ['AH' if i == 'AX' else i for i in wordPhonemes]
		length = len (wordPhonemes)
		wordOrder = list (np.arange (0, length))
		wordOrder = [i * 1.0 / length for i in wordOrder]
		return wordPhonemes, wordOrder

	def processPhonemeLine (self, phonemes):
		phonemes = phonemes.split (';')
		phonemeList = []
		orderList = []
		lenSingleList=[]
		print phonemes
		for phoneme in phonemes:
			lenSingle = len (phoneme.split (' '))
			try:
				wordPhonemes, wordOrder = self.processPhonemeSingle(phoneme)
				phonemeList += wordPhonemes
				orderList += wordOrder
				lenSingleList.append(lenSingle)
			except:
				pass
		phonemeList += ['.']
		orderList += [0.0]

		return phonemeList, orderList,lenSingleList

	def preprocessText(self, input_text):
		trans_list=[x for x in '、,，。!！?？:：;；'.decode('utf-8')]
		trans_dic={trans_list[0]:1,trans_list[1]:2,trans_list[2]:3,trans_list[3]:3,trans_list[4]:3,\
	trans_list[5]:3,trans_list[6]:3,trans_list[7]:3,trans_list[8]:3,trans_list[9]:3,trans_list[10]:3,trans_list[11]:3}
		input_text = re.sub(r'[\W]{3,50}',' ',input_text)
		input_list = []
		t_index = 0
		for i, c in enumerate(input_text):
			if c in trans_list:
				if c == ',' and re.match(r'\d,\d', input_text[i-1:]):
					continue
				input_list.append(input_text[t_index:i])
				input_list.append(trans_dic[c])
				t_index = i + 1
			elif c == '.' and re.search(r'\W{1,2}\.$',input_text[t_index:i+1]):
				s_index, e_index = re.search(r'\W{1,2}\.$',input_text[t_index:i+1]).span()
				input_list.append(input_text[t_index:s_index])
				input_list.append(3)
				t_index = i + 1

		input_list.append(input_text[t_index:])
		return input_list

	def getPhoneListwithPau(self, input_list):
		py_result = []
		for input_text in input_list:
			if type(input_text).__name__ == 'str' and len(input_text)==0:
				continue
			#print input_text
			if not type(input_text).__name__ == 'int':
				pinyin = self.getTextPhonemeTool(input_text)
				py_list = pinyin.split(';')[1:-1]
				if 'pau' in py_list:
					while 'pau' in py_list:
						py_result.append(';'.join(py_list[0:py_list.index('pau')]))
						py_result.append('pau3')
						py_list = py_list[py_list.index('pau')+1:]
					py_result.append(';'.join(py_list))
				else:
					py_result.append(';'.join(py_list))
			else:
				py_result.append('pau' + str(input_text))
			#print py_result
		return py_result



if __name__ == '__main__':
	frontTool = frontEndTool ()
	'''
	phonemeLine = frontTool.getTextPhonemeTool ("what is the weather like St. Paul, I. fear$*&^& you^^")
	phoneme, order = frontTool.processPhonemeLine(phonemeLine)
	'''
	f = open('txt','r')
	phoneme, order,_ = frontTool.text2Phoneme(f)
	#phoneme, order,_ = frontTool.text2Phoneme(["what is the weather like St. Paul, I, fear$*&^& you^^25,000"]) # I. fear$*&^& you^^25,000
	print(phoneme)
	print(order)
	f.close()
