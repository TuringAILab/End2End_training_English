#coding:utf-8
import json

with open('train_selected_N20_initial.json','r') as f:
	json_data=json.load(f)

keys=json_data.keys()
json_new={}
for i,key in enumerate(keys):
	json_new[key]=json_data[key]
	if i==4000:
		break

with open("train_selected_N20_initial_4000.json","w") as f:
	json.dump(json_new,f,indent=4)
	print("加载入文件完成...")

