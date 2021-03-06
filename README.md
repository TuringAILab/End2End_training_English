# End2End_training_English
This project is an end-to-end TTS (Speech Synthesis) system.

## 1.System requirements

python: 2.7

mxnet, numpy and other function libraries

GPU: 2×1080Ti, 12G

### Data preparation

Raw data: audio and text from LJ-speech. [Download address](https://keithito.com/LJ-Speech-Dataset/)

## 2.Data processing

1）Audio: Extract the original audio into the required features as the model output
2）Text：Extract the original audio into the required features as the model output       

The specific data processing process refers to the code in data_process.

It is divided into three steps, first run make_train_data.py, then select.py, and finally split.py.
```
   python make_train_data.py
```
```
   python select.py
```
```
   python split.py
```
The function of select.py is to filter the audio so that the length and rate are basically the same. split.py is used to filter out the training set and the validation set.

The generated **train_selected_emo_sent_data.json**, **train_selected_emo_sent_data.json** and **emo_sent_data_rate_info.npy** are put into the train_1 and trian_2 folders for training.

## 3.Model training

Model training is divided into two phases.

Phase one is mainly used to train the alignment module of the model. The training code is under the train_1 path
```
    python train.py
```
Phase two locked the alignment module, mainly training the acoustic module. The training code is under the train_2 path
```
    python train.py
```
The parameter _aco_shape_ is the feature number dimension of the generated feature file, and _aco_path_ is the address of the audio converted .pcm file

Also you have to change the .json file and npy file to the files you made in the data processing part.

## 4.audio generation

The generated code corresponding to phase one and phase two is located under the paths of synth_1 and synth_2, respectively.

The resulting audio features are the output of the model.

Input features into the synthesizer section of wav_generation / to generate the final audio.

[Paper address](https://arxiv.org/abs/1812.05710)
