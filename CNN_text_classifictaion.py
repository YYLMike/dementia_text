import numpy as np
import tensorflow as tf
import time
import os
import datetime
import csv
import matplotlib.pyplot as plt
# keras module
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
import data_preprocess
import tokenize_data_helper
import h5py
# Global variables
SEQUENCE_LENGTH = 17
EMBEDDING_DIM = 500
num_words = 1000  # word limit of tokenizer
dropout_rate = 0.7
N_FOLDS = 10
EPOCHS = 500
BATCH_SIZE = 32
message = 'CNN_dropout'
# Load Data
CONTROL_TRAIN = 'control.txt'
DEMENTIA_TRAIN = 'dementia.txt'
CONTROL_TEST = 'control_test.txt'
DEMENTIA_TEST = 'dementia_test.txt'
W2V_MODEL = '500features_20context_20mincount_zht'
# Model switches
PRE_TRAINED = True
LSTM_LAYER = True
DROPOUT_LAYER = True


def get_nn_model():
    # Architect Model
    inputs = Input(shape=(SEQUENCE_LENGTH,))
    net = inputs
    if PRE_TRAINED:
        net = Embedding(input_dim=num_words,
                        output_dim=EMBEDDING_DIM,
                        weights=[word_embedding],
                        input_length=SEQUENCE_LENGTH,
                        trainable=False)(net)
    else:
        net = Embedding(input_dim=num_words,
                        output_dim=EMBEDDING_DIM,
                        input_length=SEQUENCE_LENGTH)(net)

    pathway1 = Conv1D(kernel_size=3, strides=1, filters=64, padding='same',
                      activation='relu', name='conv_1')(net)
    pathway1 = MaxPool1D(pool_size=SEQUENCE_LENGTH)(pathway1)
    pathway2 = Conv1D(kernel_size=4, strides=1, filters=64, padding='same',
                      activation='relu', name='conv_2')(net)
    pathway2 = MaxPool1D(pool_size=SEQUENCE_LENGTH)(pathway2)
    pathway3 = Conv1D(kernel_size=5, strides=1, filters=64, padding='same',
                      activation='relu', name='conv_3')(net)
    pathway3 = MaxPool1D(pool_size=SEQUENCE_LENGTH)(pathway3)
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
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def k_fold_cross_val(n_folds):
    # K fold cross validation
    skf = StratifiedKFold(y_train, n_folds=n_folds, shuffle=True)
    best_model = None
    last_acc = 0
    acc_avg = 0
    for i, (train, val) in enumerate(skf):
        print('Running fold: ', str(i+1))
        model = get_nn_model()
        model.fit(x_train_pad[train], y_train[train],
                  epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2, callbacks=[tb])
        result = model.evaluate(x_train_pad[val], y_train[val])
        print('Validation acc: {}'.format(result[1]))
        acc_avg += result[1]
        if result[1] > last_acc:
            best_model = model
        last_acc = result[1]
    acc_avg /= n_folds
    return best_model, acc_avg

def plot_roc_curve(y_test, y_pred, out_dir):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    colors = ['aqua', 'darkorange']
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(1, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(out_dir+'roc_cure.png')

if __name__ == '__main__':
    # Load data: x_train_pad, x_test_pad, y_train, y_test
    x_train, y_train = data_preprocess.read_sentence(
        DEMENTIA_TRAIN, CONTROL_TRAIN)
    y_train = data_preprocess.label_to_scalar(y_train)
    x_train_seg = data_preprocess.segmentation(x_train)
    x_test, y_test = data_preprocess.read_sentence(DEMENTIA_TEST, CONTROL_TEST)
    y_test = data_preprocess.label_to_scalar(y_test)
    x_test_seg = data_preprocess.segmentation(x_test)
    data_helper = tokenize_data_helper.tokenize_data_helper(
        x_train_seg, num_words)
    x_train_tokens, x_test_tokens = data_helper.tokenize_data(
        x_train_seg, x_test_seg)
    x_train_pad, x_test_pad = data_helper.pad_tokenize(
        x_train_tokens, x_test_tokens)
    # Load pretrained model
    w2v_model, _, word_dict = data_preprocess.load_wordvec_model(W2V_MODEL)
    word_embedding = data_helper.embedding_matrix(word_dict, EMBEDDING_DIM)

    # Create summaries directory for Tensorboard
    timestamp = datetime.datetime.now().isoformat()
    out_dir = os.path.abspath(os.path.join(
        os.path.curdir, "runs_2", timestamp+message, "summaries"))
    tb = TensorBoard(log_dir=out_dir, histogram_freq=0,
                     write_graph=True, write_images=True)

    # Train model
    # model, acc_avg = k_fold_cross_val(N_FOLDS)
    # save_dir = os.path.join(out_dir, "..", "CNN_text.h5")
    # model.save(save_dir)
    # print('{0} Fold cross validation acc: {1:.2%}'.format(N_FOLDS, acc_avg))

    #USING restore model
    model = load_model('/home/yyliu/code/NLP/src/runs_2/2018-05-22T14:27:29.936183CNN_dropout/CNN_text.h5')
    model.summary()
    os.makedirs(out_dir)
    # Test Model
    result = model.evaluate(x_test_pad, y_test)
    print('Test set acc: {:.2%}'.format(result[1]))
    y_pred = model.predict(x=x_test_pad)

    #Plot ROC curve
    plot_roc_curve(y_test, y_pred, out_dir)
    
    # cls_pred = np.argmax(y_pred, axis=1)
    # cls_true = np.argmax(y_test, axis=1)
    
    # Save prediction csv of test set
    cls_pred = np.round(y_pred)
    cls_test = y_test
    predictions_human_readable = np.column_stack(
        (np.array(x_test_seg), cls_pred, cls_test))
    out_path = os.path.join(out_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)
