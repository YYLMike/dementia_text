# import-module
import os
import csv
import matplotlib.pyplot as plt

import jieba.posseg as pseg
import nengo
import numpy as np
from nengo.exceptions import SpaParseError, ValidationError


from tensorflow.python.keras.models import load_model

import utils.data_preprocess as data_preprocess

import h5py
import pickle
import string
SP_VOCAB = 'runs_2/SP_500epochs_500dim/'
DIM = 500
punctuation = set(string.punctuation+"，"+"、"+"」"+"「"+"。"+" "+"！")
class online_classification:

    def __init__(self):
        self.model = load_model(SP_VOCAB+'sp_30epochs_500dim.h5')

    # Use pretrained semantic pointer vocab
    def load_vocab(self, vocab_name):
        with open(vocab_name, 'rb') as f:
            self.vocab = pickle.load(f)

    def sentence2sp(self, sentence_seg_pos):
        sentence_sp = None
        new_s = self.vocab['Start']
        for word, flag in sentence_seg_pos:
            new_token = self.vocab['V' + word] * self.vocab[flag.upper()]
            new_s += new_token
        new_s.normalize()  # normalize with norm 2.
        # sentence_sp = new_s.v/(len(sentence_seg_pos)+1)# basic normalize method
        sentence_sp = new_s.v

        return sentence_sp

    # Online demo, predict keyboard input sentences
    def online_demo(self):
        while(True):
            sentence = []
            s = input('say something ... \n')
            if s == 'bye':
                break
            s_no_punc = ''
            for char in s:
                if char not in punctuation:
                    s_no_punc += char
            sentence.append(s_no_punc)

            sentence = np.array(sentence)
            # print(sentence)
            sentence_seg_pos = data_preprocess.segmentation_postagger(sentence)
            # print(sentence_seg_pos[0])
            sp = np.zeros((1, DIM))
            sp[0] = self.sentence2sp(sentence_seg_pos[0])
            predict = self.model.predict(sp)
            predict_cls = np.round(predict)
            print('\n Result: ', self.num2cls(predict_cls))
    
    def num2cls(self, num):
        if num==0:
            cls = 'Normal'
        elif num==1:
            cls = 'AD'
        return cls

    def normal_analysis(self, sentence_in):
        sentence = []
        sentence.append(sentence_in)
        sentence = np.array(sentence)
        print(sentence)
        sentence_seg_pos = data_preprocess.segmentation_postagger(sentence)
        print(sentence_seg_pos[0])
        sp = np.zeros((1, DIM))
        sp[0] = self.sentence2sp(sentence_seg_pos[0])
        predict = self.model.predict(sp)
        predict_cls = np.round(predict)
        print('\n', predict_cls)


if __name__ == '__main__':
    test_online = online_classification()
    test_online.load_vocab(SP_VOCAB+'vocab.pickle')
    test_online.online_demo()
    # s = input('say something ...')
    # test_online.normal_analysis(s)
