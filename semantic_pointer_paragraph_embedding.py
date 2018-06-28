# import-module
import os
import csv
import matplotlib.pyplot as plt

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

from tensorflow.python.keras import optimizers, regularizers
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import TensorBoard

from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.cross_validation import StratifiedKFold

import utils.data_preprocess as data_preprocess
import utils.tokenize_data_helper as tokenize_data_helper
import datetime
import h5py
import pickle

DIM = 500
TRAIN_NUM_WORDS = 1000
N_FOLDS = 10
EPOCHS = 200
BATHC_SIZE = 32
DROPOUT_RATE = 0.5
W2V_MODEL = '500features_20context_20mincount_zht'
CONTROL_TRAIN = 'control.txt'
DEMENTIA_TRAIN = 'dementia.txt'
CONTROL_TEST = 'control_test.txt'
DEMENTIA_TEST = 'dementia_test.txt'
CONTROL_TOTAL = 'control_origin.txt'
DEMENTIA_TOTAL = 'dementia_origin.txt'
SP_VOCAB = 'runs_2/SP_500epochs_500dim/'
TRAIN_MODEL = True
message = 'semantic_pointer'

class SP_sentence_embed:

    def __init__(self):
        self.w2v_model, _, self.w2v_dict = data_preprocess.load_wordvec_model(
            W2V_MODEL)
        self.load_data()

    def load_data(self):
        # self.x_train, self.y_train = data_preprocess.read_sentence(
        #     DEMENTIA_TRAIN, CONTROL_TRAIN)
        self.x_train, self.y_train = data_preprocess.read_paragraph(DEMENTIA_TRAIN, CONTROL_TRAIN)
        self.x_train_seg = data_preprocess.segmentation(self.x_train)
        self.x_train_seg_pos = data_preprocess.segmentation_postagger(
            self.x_train)

        self.x_test, self.y_test = data_preprocess.read_paragraph(
            DEMENTIA_TEST, CONTROL_TEST)
        self.x_test_seg = data_preprocess.segmentation(self.x_test)
        self.x_test_seg_pos = data_preprocess.segmentation_postagger(
            self.x_test)

        # self.data_helper = tokenize_data_helper.tokenize_data_helper(
        #     self.x_train_seg, TRAIN_NUM_WORDS)
        self.data_helper = tokenize_data_helper.tokenize_data_helper(self.x_train_seg, TRAIN_NUM_WORDS)
        self.y_train_scalar = data_preprocess.label_to_scalar(self.y_train)
        self.y_test_scalar = data_preprocess.label_to_scalar(self.y_test)

    def vocab_create(self):
        self.vocab = nengo.spa.Vocabulary(
            DIM, max_similarity=0.3)  # optional max_similarity: 0.1
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
        self.postag_flag_create()
        self.vocab_augment_postagger()
        vocab_path = os.path.join(out_dir, '..', 'vocab.pickle')
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.vocab, f, protocol=2)

    def postag_flag_create(self):
        self.flag_dict = {}
        for s in self.x_train_seg_pos:
            for word, flag in s:
                if flag not in self.flag_dict:
                    self.flag_dict[flag] = 1

    def vocab_augment_postagger(self):
        for i in self.flag_dict:
            self.vocab.parse(i.upper())

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
        new_s.normalize()# normalize with norm 2.
        # sentence_sp = new_s.v/(len(sentence_seg_pos)+1)# basic normalize method
        sentence_sp = new_s.v

        return sentence_sp

    def get_train_sentence_sp(self):
        self.x_train_sp = np.zeros((len(self.x_train_seg_pos), DIM))
        for i, s in enumerate(self.x_train_seg_pos):
            self.x_train_sp[i] = self.sentence2sp(s)
        print('get train data sentence embedding: {}'.format(len(self.x_train_sp)))

    def get_test_sentnece_sp(self):
        self.x_test_sp = np.zeros((len(self.x_test_seg_pos), DIM))
        for i, s in enumerate(self.x_test_seg_pos):
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
                i, self.x_train_sp, self.y_train_scalar, cv=N_FOLDS, scoring=scoring, return_train_score=True)
            for k in scores.keys():
                print(str(n) + str(k) +
                      '\nscore: {}'.format(np.mean(scores[k])))
            print('-'*10)
            # i.fit(self.x_train_sp, self.y_train_scalar)
            # print(i.predict(self.x_test_sp))
            # print('Test score: {}'.format(
            #     i.score(self.x_test_sp, self.y_test_scalar)))
            # print('-'*30)
        return decision_tree

    def nn_model(self):
        model = Sequential()
        layer_dim = [250, 50, 1]  # 256 32 1
        model.add(Dropout(rate=DROPOUT_RATE, input_shape=(DIM,)))
        model.add(Dense(layer_dim[0], activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(Dense(layer_dim[1], activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(rate=DROPOUT_RATE))
        model.add(Dense(layer_dim[2], activation='sigmoid'))
        optimizer = optimizers.Adam()
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def k_cross_val(self, n_folds, tb):

        skf = StratifiedKFold(self.y_train_scalar, n_folds=n_folds, shuffle=True)

        best_model = None
        last_acc = 0
        acc_avg = 0
        for i, (train, val) in enumerate(skf):
            print('Running fold: ', str(i+1))
            model = self.nn_model()
            model.fit(self.x_train_sp[train], self.y_train_scalar[train],
                        epochs=EPOCHS, batch_size=BATHC_SIZE, shuffle='True', verbose=2, callbacks=[tb])
            result = model.evaluate(
                self.x_train_sp[val], self.y_train_scalar[val])
            print('Validation acc: {}'.format(result[1]))
            if result[1] > last_acc:
                best_model = model
                y_pred = model.predict(self.x_train_sp[val])
                self.plot_roc_curve(self.y_train_scalar[val], y_pred)
            last_acc = result[1]
            acc_avg += last_acc
        acc_avg /= N_FOLDS
        return best_model, acc_avg

    def plot_roc_curve(self, y_test, y_pred):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        fpr, tpr, threshold = roc_curve(y_test, y_pred)
        # print('threshold: ', threshold)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2

        plt.plot(fpr, tpr, color='aqua', lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(1, roc_auc))
        plt.plot([0, 1],[0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        # plt.show()
        plot_path = os.path.join(out_dir, '..', 'roc_curve.png')
        plt.savefig(plot_path)

    def create_prediction_csv(self, y_pred):
        cls_pred = np.round(y_pred)
        predictions_human_readable = np.column_stack(
        (np.array(self.x_test_seg), cls_pred, self.y_test_scalar))
        out_path = os.path.join(SP_VOCAB, "prediction.csv")
        print("Saving evaluation to {0}".format(out_path))
        with open(out_path, 'w') as f:
            csv.writer(f).writerows(predictions_human_readable)

    # Online demo, predict keyboard input sentences
    def online_demo(self):
        while(True):
            sentence = []
            s = input('say something ... ')
            if s=='bye':
                break
            sentence.append(s)
            sentence = np.array(sentence)
            print(sentence)
            sentence_seg_pos = data_preprocess.segmentation_postagger(sentence)
            print(sentence_seg_pos[0])
            sp = np.zeros((1,DIM))
            sp[0] = self.sentence2sp(sentence_seg_pos[0])
            predict = model.predict(sp)
            predict_cls = np.round(predict
)
            print('\n', predict_cls)
        out_dir
    
        
if __name__ == '__main__':
    timestamp = datetime.datetime.now().isoformat()
    global out_dir
    out_dir = os.path.abspath(os.path.join(
        os.path.curdir, "runs_2", timestamp+message, "summaries"))
    os.makedirs(out_dir)
    tb = TensorBoard(log_dir=out_dir, histogram_freq=0,
                     write_graph=True, write_images=True)
    test_sp_s2v = SP_sentence_embed()
    
    if TRAIN_MODEL:
        test_sp_s2v.vocab_create()
    else:
        test_sp_s2v.load_vocab(SP_VOCAB + 'vocab.pickle')
    
    test_sp_s2v.get_train_sentence_sp()
    test_sp_s2v.get_test_sentnece_sp()
    ## Use basic ml mehtod
    # decicsion_tree = test_sp_s2v.evaluate_with_ml()
    ######

    ## Train model
    # if TRAIN_MODEL:
    #     model, val_acc_avg = test_sp_s2v.k_cross_val(N_FOLDS, tb)
    #     model.save('sp_30epochs_500dim.h5', include_optimizer=False)
    #     print('{0} fold cross validation acc: {1:.2f}'.format(N_FOLDS, val_acc_avg))
    if TRAIN_MODEL:
        model = test_sp_s2v.nn_model()
        model.fit(test_sp_s2v.x_train_sp, test_sp_s2v.y_train_scalar, epochs=EPOCHS, batch_size=BATHC_SIZE, shuffle='True', verbose=2, callbacks=[tb])
        model_path = os.path.join(out_dir,'..', 'sp_30epochs_500dim.h5')
        model.save(model_path, include_optimizer=False)
    #######
    # ## Use restore model 
    # else:
    #     model = load_model(SP_VOCAB + 'sp_30epochs_500dim.h5')
    num_cls = 1
    y_pred = model.predict(test_sp_s2v.x_test_sp)
    result = model.evaluate(test_sp_s2v.x_test_sp, test_sp_s2v.y_test_scalar)
    print('test acc: {:.2f}'.format(result[1]))

    test_sp_s2v.plot_roc_curve(test_sp_s2v.y_test_scalar, y_pred)
    test_sp_s2v.create_prediction_csv(y_pred)
    # test_sp_s2v.online_demo()
    
