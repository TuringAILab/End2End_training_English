# End2End_training_English
本工程是一个端到端的TTS（语音合成）系统。
一、系统要求

python： 2.7

mxnet，numpy等函数库

GPU：1080Ti， 12G

二、数据准备

原始数据：LJ-speech的音频和文本。

数据处理：1）音频：将原始音频提取为需要的特征，作为模型输出

          2）文本：将文本转换为CMU音标和其他特征。

具体的数据处理过程参考data_process内的代码。

三、模型训练

模型训练分为两个阶段。

阶段一主要用于训练模型的alignment模块。训练代码在 train_1 路径下

阶段二锁定了alignment模块，主要训练acoustic模块。训练代码在 train_2 路径下

四、音频生成

对应于阶段一和阶段二的生成代码，分别位于synth_1和synth_2的路径下。

生成结果为模型的输出的音频特征。

将特征输入wav_generation/的合成器部分， 生成最后的音频。
