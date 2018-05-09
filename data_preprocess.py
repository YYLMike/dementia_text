# import modules
import pandas as pd
import pickle
import gensim
import numpy as np
import string
from opencc import OpenCC
import ckip
import jieba
import tensorflow as tf
# Path of files
CSV_DATA = '../data/CookieTheft_51.csv'
DEMENTIA_DATA = '../data/'
CONTROL_DATA = '../data/'
WORDVEC_MODEL = '../wordvec_model/'
# Variables
DEMENTIA_NUM = 51
CONTROL_NUM = 51
WV_DIIM = 500
INDEX_CONTROL_START = 68  # The end of dementia id is 67
JIEBA_DICT = '../data/dict.txt.big'
jieba.set_dictionary(JIEBA_DICT)

punctuation = set(string.punctuation+"，"+"、"+"」"+"「"+"。"+" "+"！")

# csv file of control subjects, transform to txt file


def csv_to_txt(file_name):
    csv = pd.read_csv(CSV_DATA, header=None)
    with open(file_name, 'w', encoding='utf8') as f:
        idx = INDEX_CONTROL_START
        for line in csv.iloc[1:, 1]:
            f.write(str(idx) + '\n')
            f.write(line + '\n')
            idx += 1
    print('Control subject CSV file to txt sucess ...')

# read both dementia&control txt file into paragraph dictionary, size: 102


def read_paragraph(file_name=None):
    paragraph = []
    paragraph_dict = {}
    with open(DEMENTIA_DATA, encoding='utf8') as f:
        paragraph = f.readlines()
    for i in range(len(paragraph)):
        if i % 2 == 0:
            paragraph_dict[paragraph[i].strip(
                '\n')] = paragraph[i+1].strip('\n')
    with open(CONTROL_DATA, encoding='utf8') as f:
        paragraph = f.readlines()
    for i in range(len(paragraph)):
        if i % 2 == 0:
            paragraph_dict[paragraph[i].strip(
                '\n')] = paragraph[i+1].strip('\n')
    if file_name:
        with open(file_name, 'wb') as f:
            pickle.dump(paragraph_dict, f)
    print(len(paragraph_dict))
    print('read_paragraph done ...')
    return paragraph_dict

# r read both dementia&control txt file into sentence list, size: 873


def read_sentence(file_name_ad=None, file_name_ctrl=None):
    sentence = []
    with open(DEMENTIA_DATA+file_name_ad, encoding='utf8') as f:
        dementia_txt = f.readlines()
    for i in range(len(dementia_txt)):
        if i % 2 == 0:
            sentence.extend(split_punctuation(dementia_txt[i+1]))
    dementia_num = len(sentence)
    with open(CONTROL_DATA+file_name_ctrl, encoding='utf8') as f:
        control_txt = f.readlines()
    for i in range(len(control_txt)):
        if i % 2 == 0:
            sentence.extend(split_punctuation(control_txt[i+1]))
    control_num = len(sentence) - dementia_num
    train_data = np.array(sentence)
    dementia_labels = [[0, 1] for _ in train_data[:dementia_num]]
    control_labels = [[1, 0] for _ in train_data[dementia_num:]]
    train_labels = np.concatenate([dementia_labels, control_labels], 0)
    print('total number of train set: {}'.format(train_data.shape[0]))
    print('sentence number of dementia subject: {}'.format(len(dementia_labels)))
    print('sentence number of control normal subject: {}'.format(len(control_labels)))
    return train_data, train_labels

# segment sentence into token list


def segmentation(train_data):
    train_data_seg = []
    for i in train_data:
        train_data_seg.append(' '.join(jieba.lcut(i)))
    return train_data_seg

# split paragrpah to sentence by punctuation


def split_punctuation(sentence):
    sentence_split = []
    tmp = ''
    for i in sentence:
        if i not in punctuation:
            tmp += i
        else:
            sentence_split.append(tmp)
            tmp = ''
    return sentence_split

# load word2vec model


def load_wordvec_model(file_name):
    w2v_model = gensim.models.Word2Vec.load(WORDVEC_MODEL+file_name)
    words = []
    for word in w2v_model.wv.vocab:
        words.append(word)
    word_dict = {}
    for k in w2v_model.wv.vocab.keys():
        word_dict[k] = np.asarray(w2v_model.wv[k], dtype='float32')
    word_embedding = []
    for k in w2v_model.wv.vocab.keys():
        word_embedding.append(np.asarray(w2v_model.wv[k]))
    print('Load word2vec model sucess ...')
    print('Number of token: {}'.format(len(words)))
    print('Dimensions of word vector: {}'.format(len(w2v_model[words[0]])))
    return w2v_model, word_embedding, word_dict

# sentence to sequence by onehot encoding


def text_to_onehot(train_data_seg):
    # max_sentence_length = max([len(x.split(' ')) for x in train_data_seg])
    max_sentence_length = 17
    print('Max token number of sentence: {}'.format(max_sentence_length))
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_sentence_length)
    x_onehot = np.array(list(vocab_processor.fit_transform(train_data_seg)))
    x_onehot = np.asarray(train_data_seg_encoded, dtype=np.int32)
    print('Sample of x_onehot: {}'.format(x_onehot[-1]))
    print('Vocabulary size: {}'.format(len(vocab_processor.vocabulary_)))

    return x_onehot, vocab_processor


def text_to_onehot_w2v(train_data_seg, vocab):
    # max_sentence_length = max([len(x.split(' ')) for x in train_data_seg])
    max_sentence_length = 17
    print('Max token number of sentence: {}'.format(max_sentence_length))
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_sentence_length)
    vocab_processor.fit(vocab)
    x_onehot_w2v = np.array(list(vocab_processor.transform(train_data_seg)))
    print('Sample of x_onehot_w2v: {}'.format(x_onehot_w2v[-1]))
    print('Vocabulary size: {}'.format(len(vocab_processor.vocabulary_)))
    return x_onehot_w2v, vocab_processor


def cross_validate_data(data_train, data_label):
    dev_sample_index = -1 * int(.1*float(len(data_label)))
    x_train, x_dev = data_train[:dev_sample_index], data_train[dev_sample_index:]
    y_train, y_dev = data_label[:dev_sample_index], data_label[dev_sample_index:]
    print('Train/Dev split : {}/{}'.format(len(y_train), len(y_dev)))
    return x_train, x_dev, y_train, y_dev


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    pass
    # csv_to_txt('control.txt')
    # sentence_dict = read_paragraph()
    # sentence = '3個人，一個媽媽兩個小孩，小孩站在椅子上要拿西點，椅子都快倒下來了，在拿這個西點餅乾要吃，手下還拿著一塊，'
    # print(split_punctuation(sentence))
    # read_paragraph()
    # x, y = read_sentence()
    # x_seg = segmentation(x)
    # w2v_model = load_wordvec_model('500features_20context_20mincount')
    # x_onehot = text_to_onehot(x_seg)
