# import-module
import os
import csv
import matplotlib.pyplot as plt

import jieba.posseg as pseg
import nengo
import numpy as np
from nengo.exceptions import SpaParseError, ValidationError


from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import utils.data_preprocess as data_preprocess
import utils.tokenize_data_helper as data_helper
import h5py
import pickle

# MODEL_DIR = 'runs_2/2018-06-06T13:53:01.677359CNN_dropout_paragrpah/'
MODEL_DIR = 'runs_2/2018-06-28T14:45:11.292488CNN_dropout/'
CNN_DATA_HELPER = '../pickle/data_helper_CNN.pickle'
DIM = 500
# with open(CNN_DATA_HELPER, 'rb') as f:
#     data_helper = pickle.load(f)
with open('tokenizer_nfull.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
class online_classification:

    def __init__(self):
        self.model = load_model(MODEL_DIR+'CNN_text.h5')
        self.sequence_length = self.model.input[0].shape[0].value
    # Use pretrained semantic pointer vocab
    def load_vocab(self, vocab_name):
        with open(vocab_name, 'rb') as f:
            self.vocab = pickle.load(f)

    def sentence2sp(self, sentence_seg_pos):
        sentence_sp = None
        new_s = self.vocab['Start']
        for word, flag in sentence_seg_pos:
            new_token = self.vocab['V' + str(word)] * self.vocab[flag.upper()]
            new_s += new_token
        new_s.normalize()  # normalize with norm 2.
        # sentence_sp = new_s.v/(len(sentence_seg_pos)+1)# basic normalize method
        sentence_sp = new_s.v

        return sentence_sp

    # Online demo, predict keyboard input sentences
    def online_demo(self):
        while(True):
            sentence = []
            s = input('say something ... ')
            if s == 'bye':
                break
            sentence.append(s)
            sentence = np.array(sentence)
            print(sentence)
            sentence_seg = data_preprocess.segmentation(sentence)
            print(sentence_seg[0])
            # x_train_tokens = data_helper.tokenize_data(sentence_seg)
            x_train_tokens = tokenizer.texts_to_sequences(sentence_seg)
            print(x_train_tokens)
            # x_train_pad = data_helper.pad_tokenize(x_train_tokens, maxlen=self.sequence_length)
            x_train_pad = pad_sequences(x_train_tokens, maxlen=self.sequence_length, padding='post', truncating='post')
            print(x_train_pad)
            predict = self.model.predict(x_train_pad)
            predict_cls = np.round(predict)
            print('\n', predict_cls)

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
    # test_online.load_vocab(SP_VOCAB+'vocab.pickle')
    test_online.online_demo()
    # s = input('say something ...')
    # test_online.normal_analysis(s)
