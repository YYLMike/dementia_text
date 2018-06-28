# import module
import numpy as np
import tensorflow as tf
import time
import os
import datetime
import csv
import h5py
import pickle
import matplotlib.pyplot as plt
# tensorflow keras module
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, LSTM, Embedding, Input, Conv1D, MaxPool1D, concatenate, Flatten, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import TensorBoard
#sklearn module
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.cross_validation import StratifiedKFold

# data preprocess module
import utils.data_preprocess as data_preprocess
import utils.tokenize_data_helper as tokenize_data_helper

# Global variables
SEQ_LEN_PARAGRAPH = 200 # paragraph level
SEQ_LEN_SENTENCE = 17 # sentence level
EMBEDDING_DIM = 500
NUM_WORDS = 1000  # word limit of tokenizer
dropout_rate = 0.5

N_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 32

# Load Data
CONTROL_TRAIN = 'control.txt'
DEMENTIA_TRAIN = 'dementia.txt'
CONTROL_TEST = 'control_test.txt'
DEMENTIA_TEST = 'dementia_test.txt'
CONTROL_TOTAL = 'control_origin.txt'
DEMENTIA_TOTAL = 'dementia_origin.txt'
W2V_MODEL = '500features_20context_20mincount_zht'
# Model switches
PRE_TRAINED = True
LSTM_LAYER = True
DROPOUT_LAYER = True


class CNN_text_classifier:
    
    def __init__(self, text_level, cv=False):
        
        if text_level=='sentence':
            self.text_level = 'sentence'
            self.SEQ_LEN = SEQ_LEN_SENTENCE
        elif text_level=='paragraph':
            self.text_level = 'paragraph'
            self.SEQ_LEN = SEQ_LEN_PARAGRAPH            
        self.embedding_matrix = None
        self.cv = cv

    def get_nn_model(self):
        inputs = Input(shape=(self.SEQ_LEN,))
        net = inputs

        if PRE_TRAINED:
            net = Embedding(input_dim=NUM_WORDS,
                            output_dim=EMBEDDING_DIM,
                            weights=[self.embedding_matrix],
                            input_length=self.SEQ_LEN,
                            trainable=False)(net)
        else:
            net = Embedding(input_dim=NUM_WORDS,
                            output_dim=EMBEDDING_DIM,
                            input_length=self.SEQ_LEN)(net)
        net = Dropout(rate=dropout_rate)(net)
        pathway1 = Conv1D(kernel_size=3, strides=1, filters=64, padding='same',
                        activation='relu', name='conv_1')(net)
        pathway1 = MaxPool1D(pool_size=self.SEQ_LEN)(pathway1)
        pathway2 = Conv1D(kernel_size=4, strides=1, filters=64, padding='same',
                        activation='relu', name='conv_2')(net)
        pathway2 = MaxPool1D(pool_size=self.SEQ_LEN)(pathway2)
        pathway3 = Conv1D(kernel_size=5, strides=1, filters=64, padding='same',
                        activation='relu', name='conv_3')(net)
        pathway3 = MaxPool1D(pool_size=self.SEQ_LEN)(pathway3)
        net = concatenate([pathway1, pathway2, pathway3], axis=2)
        if DROPOUT_LAYER:
            net = Dropout(rate=dropout_rate)(net)
        if LSTM_LAYER:
            if DROPOUT_LAYER:
                net = LSTM(units=32, return_sequences=True,
                        name='LSTM_1', dropout=dropout_rate)(net)
                net = LSTM(units=8, name='LSTM_2', dropout=dropout_rate)(net)
            else:
                net = LSTM(units=32, return_sequences=True, name='LSTM_1')(net)
                net = LSTM(units=8, name='LSTM_2')(net)

        net = Dense(1, activation='sigmoid')(net)
        net = Flatten()(net)
        outputs = net
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='Adam', loss='binary_crossentropy',
                    metrics=['accuracy'])
        return self.model
    
    def k_fold_cross_val(self, x_train_pad, y_train, word_embedding, n_folds, tb):
        # K fold cross validation
        skf = StratifiedKFold(y_train, n_folds=n_folds, shuffle=True)
        best_model = None
        last_acc = 0
        acc_avg = 0

        for i, (train, val) in enumerate(skf):
            print('Running fold: ', str(i+1))
            self.get_nn_model()
            self.model.fit(x_train_pad[train], y_train[train],
                    epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tb])
            result = self.model.evaluate(x_train_pad[val], y_train[val])

            print('Validation acc: {}'.format(result[1]))
            acc_avg += result[1]
            if result[1] > last_acc:
                best_model = self.model
                y_pred = self.model.predict(x_train_pad[val])
                self.plot_roc_curve(y_train[val], y_pred, self.out_dir)
            last_acc = result[1]
        acc_avg /= n_folds
        return best_model, acc_avg
    
    def plot_roc_curve(self, y_test, y_pred, out_dir):
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(1, roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plot_path = os.path.join(out_dir, '..', 'roc_curve.png')
        plt.savefig(plot_path)
    
    def load_data_total(self):
        if self.text_level=='sentence':
            self.x_train, self.y_train = data_preprocess.read_sentence(DEMENTIA_TOTAL, CONTROL_TOTAL)
        elif self.text_level=='paragraph':
            self.x_train, self.y_train = data_preprocess.read_paragraph(DEMENTIA_TOTAL, CONTROL_TOTAL)

        self.y_train = data_preprocess.label_to_scalar(self.y_train)
        self.x_train_seg = data_preprocess.segmentation(self.x_train)
        self.data_helper = tokenize_data_helper.tokenize_data_helper(self.x_train_seg, NUM_WORDS)
        self.x_train_tokens = self.data_helper.tokenize_data(self.x_train_seg)
        self.x_train_pad = self.data_helper.pad_tokenize(x_train_tokens=self.x_train_tokens, maxlen=self.SEQ_LEN)
        print('load cv data done ...\n')
        
    def load_data(self):
        if self.text_level=='sentence':
            self.x_train, self.y_train = data_preprocess.read_sentence(DEMENTIA_TRAIN, CONTROL_TRAIN)
            self.x_test, self.y_test = data_preprocess.read_sentence(DEMENTIA_TEST, CONTROL_TEST)
        elif self.text_level=='paragraph':
            self.x_train, self.y_train = data_preprocess.read_paragraph(DEMENTIA_TRAIN, CONTROL_TRAIN)  
            self.x_test, self.y_test = data_preprocess.read_paragraph(DEMENTIA_TEST, CONTROL_TEST)
       
        self.y_train = data_preprocess.label_to_scalar(self.y_train)
        self.x_train_seg = data_preprocess.segmentation(self.x_train)
        self.y_test = data_preprocess.label_to_scalar(self.y_test)
        self.x_test_seg = data_preprocess.segmentation(self.x_test)

        self.data_helper = tokenize_data_helper.tokenize_data_helper(self.x_train_seg, NUM_WORDS)
        self.x_train_tokens, self.x_test_tokens = self.data_helper.tokenize_data(self.x_train_seg, self.x_test_seg)
        self.x_train_pad, self.x_test_pad = self.data_helper.pad_tokenize(x_train_tokens=self.x_train_tokens, x_test_tokens=self.x_test_tokens, maxlen=self.SEQ_LEN)
        print('load data done ...\n')

    def load_embedding_matrix(self):
        w2v_model, _, word_dict = data_preprocess.load_wordvec_model(W2V_MODEL)
        self.embedding_matrix = self.data_helper.embedding_matrix(word_dict, EMBEDDING_DIM)
        print('load embedding matrix done ...\n')

    def train_model(self):
        timestamp = datetime.datetime.now().isoformat()
        file_name = 'CNN_text_classification_' + self.text_level
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_2", timestamp+file_name, "summaries"))
        tb = TensorBoard(log_dir=self.out_dir, histogram_freq=0, write_graph=True, write_images=True)
        if self.cv==True:
            self.model, acc_avg = self.k_fold_cross_val(self.x_train_pad, self.y_train, self.embedding_matrix, N_FOLDS, tb)
            print('{0} Fold cross validation acc: {1:.2%}'.format(N_FOLDS, acc_avg))
        else:
            self.get_nn_model()
            self.model.fit(self.x_train_pad, self.y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tb])
            result = self.model.evaluate(self.x_test_pad, self.y_test)
            print('Test acc: {:.2%}'.format(result[1]))
        print('train model done ...\n')

    def test_model(self):
        y_pred = self.model.predict(x=self.x_test_pad)
        self.plot_roc_curve(self.y_test, y_pred, self.out_dir)
        cls_pred = np.round(y_pred)
        cls_test = self.y_test
        predictions_human_readable = np.column_stack((np.array(self.x_test_seg), cls_pred, cls_test))
        out_path = os.path.join(self.out_dir, "..", "prediction.csv")
        with open(out_path, 'w') as f:
            csv.writer(f).writerows(predictions_human_readable)
        print('test model done ...\n')
    
    def save_model(self):
        save_dir = os.path.join(self.out_dir, "..", "CNN_text.h5")
        self.model.save(save_dir)
        print('save model done ...\n')

if __name__ == '__main__':
    pass
