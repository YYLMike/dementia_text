import numpy as np
import tensorflow as tf
import time
import os
import datetime
import csv

# keras module
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, LSTM, Embedding, Input, Conv1D, MaxPool1D, concatenate, Flatten, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import TensorBoard

# data preprocess module
import data_preprocess
import tokenize_data_helper
# global variables
SEQUENCE_LENGTH = 17
EMBEDDING_DIM = 100
num_words = 1000 # word limit of tokenizer
CONTROL_TRAIN = 'control.txt'
DEMENTIA_TRAIN = 'dementia.txt'
CONTROL_TEST = 'control_test.txt'
DEMENTIA_TEST = 'dementia_test.txt'
W2V_MODEL = '100features_20context_20mincount_zht'
EPOCHS = 500
BATCH_SIZE = 32
message = 'CNN_dropout'
PRE_TRAINED = False
LSTM_LAYER = False
DROPOUT_LAYER = True
dropout_rate = 0.7

if __name__ =='__main__':
    # Load data: x_train_pad, x_test_pad, y_train, y_test
    x_train, y_train = data_preprocess.read_sentence(DEMENTIA_TRAIN, CONTROL_TRAIN)
    x_train_seg = data_preprocess.segmentation(x_train)
    x_test, y_test = data_preprocess.read_sentence(DEMENTIA_TEST, CONTROL_TEST)
    x_test_seg = data_preprocess.segmentation(x_test)
    data_helper = tokenize_data_helper.tokenize_data_helper(x_train_seg, num_words)
    x_train_tokens, x_test_tokens = data_helper.tokenize_data(x_train_seg, x_test_seg)
    x_train_pad, x_test_pad = data_helper.pad_tokenize(x_train_tokens, x_test_tokens)
    # Load pretrained model
    w2v_model, _, word_dict = data_preprocess.load_wordvec_model(W2V_MODEL)
    word_embedding = data_helper.embedding_matrix(word_dict, EMBEDDING_DIM)

    # Create summaries directory for Tensorboard
    timestamp = datetime.datetime.now().isoformat()
    out_dir = os.path.abspath(os.path.join(
                os.path.curdir, "runs_2", timestamp+message, "summaries"))
    tb = TensorBoard(log_dir=out_dir, histogram_freq=0, write_graph=True, write_images=True)

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
            net = LSTM(units=32, return_sequences=True, name='LSTM_1', dropout=dropout_rate)(net)
            net = LSTM(units=8, name='LSTM_2', dropout=dropout_rate)(net)
        else:
            net = LSTM(units=32, return_sequences=True, name='LSTM_1')(net)
            net = LSTM(units=8, name='LSTM_2')(net)

    net = Dense(2, activation='sigmoid')(net)
    net = Flatten()(net)
    outputs = net
    model = Model(inputs=inputs, outputs=outputs)

    # Train Model
    model.summary()
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train_pad, y_train,
            validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, 
            shuffle=True, 
            callbacks=[tb])
    
    # Test Model
    result = model.evaluate(x_test_pad, y_test)
    print('Accuracy: {:.2%}'.format(result[1]))
    
    y_pred = model.predict(x=x_test_pad)
    cls_pred = np.argmax(y_pred, axis=1)
    cls_true = np.argmax(y_test, axis=1)

    predictions_human_readable = np.column_stack((np.array(x_test_seg), cls_pred, cls_true))
    out_path = os.path.join(out_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)
    