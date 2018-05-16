# import-module
import sys

import jieba.posseg as pseg
import nengo
import numpy as np
from nengo.exceptions import SpaParseError, ValidationError
from nengo.spa import pointer

from sklearn.model_selection import cross_validate
from sklearn import tree  # Decision Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

import data_preprocess
import tokenize_data_helper

DIM = 100
TRAIN_NUM_WORDS = 1000
W2V_MODEL = '100features_20context_20mincount_zht'
CONTROL_TRAIN = 'control.txt'
DEMENTIA_TRAIN = 'dementia.txt'
CONTROL_TEST = 'control_test.txt'
DEMENTIA_TEST = 'dementia_test.txt'


class SP_sentence_embed:

    def __init__(self):
        self.w2v_model, _, self.w2v_dict = data_preprocess.load_wordvec_model(
            W2V_MODEL)
        self.vocab = nengo.spa.Vocabulary(
            DIM, max_similarity=0.3)  # optional max_similarity: 0.1
        self.load_data()

    def load_data(self):
        self.x_train, self.y_train = data_preprocess.read_sentence(
            DEMENTIA_TRAIN, CONTROL_TRAIN)
        self.x_train_seg = data_preprocess.segmentation(self.x_train)
        self.x_train_seg_pos = data_preprocess.segmentation_postagger(
            self.x_train)

        self.x_test, self.y_test = data_preprocess.read_sentence(
            DEMENTIA_TEST, CONTROL_TEST)
        self.test_seg = data_preprocess.segmentation(self.x_test)
        self.x_test_seg_pos = data_preprocess.segmentation_postagger(
            self.x_test)

        self.data_helper = tokenize_data_helper.tokenize_data_helper(
            self.x_train_seg, TRAIN_NUM_WORDS)

        self.y_train_scalar = data_preprocess.label_to_scalar(self.y_train)
        self.y_test_scalar = data_preprocess.label_to_scalar(self.y_test)

    def vocab_create(self):
        self.oov = []
        for token, i in self.data_helper.tokenizer.word_index.items():
            try:
                self.vocab.add(str('V'+token), self.w2v_dict[token])
            except KeyError:
                self.oov.append(token)
                value = self.vocab.create_pointer(attempts=100)
                self.vocab.add(str('V'+token), value)
                continue
        self.vocab.add('Start', np.zeros(DIM))
        print('oov number: {}'.format(len(self.oov)))
        assert len(self.vocab.keys) - \
            1 == len(self.data_helper.tokenizer.word_index.keys())
        print('spa vocab size: {}'.format(len(self.vocab.keys)))
        print('tokenizer word index size: {}'.format(
            len(self.data_helper.tokenizer.word_index.keys())))
        self.postag_falg_create()
        self.vocab_augment_postagger()

    def postag_falg_create(self):
        self.flag_dict = {}
        for s in self.x_train_seg_pos:
            for word, flag in s:
                if flag not in self.flag_dict:
                    self.flag_dict[flag] = 1

    def vocab_augment_postagger(self):
        for i in self.flag_dict:
            self.vocab.parse(i.upper())

    # def sentence_sp(self, sentence):
    #     sentence_bind_pos = []
    #     for s in self.x_train_seg_pos:
    #         new_s = self.vocab['Start']
    #         for word, flag in s:
    #             new_token = self.vocab['V' +
    #                                    str(word)] * self.vocab[flag.upper()]
    #             new_s += new_token
    #         sentence_bind_pos.append(new_s.v/len(s))  # normalize
    #     self.x_train_sp = np.zeros((len(self.x_train), DIM))
    #     for i in range(len(sentence_bind_pos)):
    #         self.x_train_sp[i] = sentence_bind_pos[i]
    #     print('sentence embedding using semantic pointer architecture ...')
    
    def sentence2sp(self, sentence_seg_pos):
        sentence_sp = None
        new_s = self.vocab['Start']
        for word, flag in sentence_seg_pos:
            new_token = self.vocab['V' + str(word)] * self.vocab[flag.upper()]
            new_s += new_token
        sentence_sp = new_s.v/len(sentence_seg_pos)
        return sentence_sp
    
    def get_train_sentence_sp(self):
        self.x_train_sp = np.zeros((len(self.x_train_seg_pos), DIM))
        for i,s in enumerate(self.x_train_seg_pos):
            self.x_train_sp[i] = self.sentence2sp(s)
        print('get train data sentence embedding: {}'.format(len(self.x_train_sp)))
    
    def get_test_sentnece_sp(self):
        self.x_test_sp = np.zeros((len(self.x_test_seg_pos), DIM))
        for i,s in enumerate(self.x_test_seg_pos):
            self.x_test_sp[i] = self.sentence2sp(s)
        print('get test sentence embedding: {}'.format(len(self.x_test_sp)))

    def evaluate_with_ml(self):

        decision_tree = tree.DecisionTreeClassifier()
        forest = RandomForestClassifier()
        logistic = LogisticRegression()
        svm_classifier = SVC()
        knn = KNeighborsClassifier(n_neighbors=10)
        
        ml_method_name = ['decision_tree', 'random forest',
                          'logidtic regression', 'svm', 'knn']
        ml_method = (decision_tree, forest, logistic,
                     svm_classifier, knn)
        scoring = ['f1_micro', 'accuracy']
        for i, n in zip(ml_method, ml_method_name):
            scores = cross_validate(
                i, self.x_train_sp, self.y_train_scalar, cv=10, scoring=scoring, return_train_score=True)
            for k in scores.keys():
                print(str(n) + str(k) + '\nscore: {}'.format(np.mean(scores[k])))
            print('-'*10)
            i.fit(self.x_train_sp, self.y_train_scalar)
            print(i.predict(self.x_test_sp))
            print('Test score: {}'.format(i.score(self.x_test_sp, self.y_test_scalar)))
            print('-'*30)
    
    def nn_model(self):
        self.model = Sequential()
        layer_dim = [256, 16, 1]# 256 32 1
        self.model.add(Dense(layer_dim[0], input_dim=DIM, activation='relu'))
        self.model.add(Dense(layer_dim[1], activation='relu'))
        self.model.add(Dense(layer_dim[2], activation='sigmoid'))
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train_sp, self.y_train_scalar, validation_split=0.1, epochs=50, batch_size=32, shuffle='True')
    
if __name__ == '__main__':
    test_sp_s2v = SP_sentence_embed()
    test_sp_s2v.vocab_create()
    test_sp_s2v.get_train_sentence_sp()
    test_sp_s2v.get_test_sentnece_sp()
    test_sp_s2v.evaluate_with_ml()
    # test_sp_s2v.nn_model()
