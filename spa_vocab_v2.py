# import module
import data_preprocess
import nengo
from nengo.exceptions import SpaParseError, ValidationError
import sys
import numpy as np
from nengo.spa import pointer
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

class Data_helper():

    def __init__(self, fit_data, num_words):
        self.tokenizer = self.tokenize(fit_data, num_words)
        self.idx = self.tokenizer.word_index
        self.inverse_map = dict(zip(self.idx.values(), self.idx.keys()))
    
    def tokenize(self, fit_data, num_words=1000):
        self.tokenizer = Tokenizer(num_words=num_words)
        self.tokenizer.fit_on_texts(fit_data)
        return self.tokenizer

    def tokenize_data(self, x_train, x_test):
        return self.tokenizer.texts_to_sequences(x_train), self.tokenizer.texts_to_sequences(x_test)

    def pad_tokenize(self, x_train_tokens, x_test_tokens, pad='post'):
        num_tokens = [len(tokens) for tokens in x_train_tokens+x_test_tokens]
        global max_tokens
        max_tokens = np.max(num_tokens)
        x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens, padding=pad, truncating=pad)
        x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens, padding=pad, truncating=pad)
        return x_train_pad, x_test_pad

    def tokens_to_string(self, tokens):
        words = [self.inverse_map[token] for token in tokens if token != 0]
        text = ' '.join(words)
        return text
    
    def embedding_matrix(self, pretrained_dict):
        word_index = self.tokenizer.word_index
        word_embedding = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = pretrained_dict.get(word)
            if embedding_vector is not None:
                word_embedding[i] = embedding_vector
        return word_embedding


DIM = 100
W2V_MODEL = '100features_20context_20mincount_zht'
w2v_model, _, w2v_dict = data_preprocess.load_wordvec_model(W2V_MODEL)
vocab = nengo.spa.Vocabulary(100, max_similarity=0.3) # optional max_similarity: 0.1

CONTROL_TRAIN = 'control.txt'
DEMENTIA_TRAIN = 'dementia.txt'
CONTROL_TEST = 'control_test.txt'
DEMENTIA_TEST = 'dementia_test.txt'
x_train, y_train = data_preprocess.read_sentence(DEMENTIA_TRAIN, CONTROL_TRAIN)
x_train_seg = data_preprocess.segmentation(x_train)
x_test, y_test = data_preprocess.read_sentence(DEMENTIA_TEST, CONTROL_TEST)
x_test_seg = data_preprocess.segmentation(x_test)
data_helper = Data_helper(x_train_seg, num_words)
x_train_tokens, x_test_tokens = data_helper.tokenize_data(x_train_seg, x_test_seg)
x_train_pad, x_test_pad = data_helper.pad_tokenize(x_train_tokens, x_test_tokens)

oov = []
for sentence in x_train_seg:
    for word in sentence.split():
        try:
            vocab.add(word, w2v_dict[word])
        except KeyError:
            oov.append(word)
            value = vocab.create_pointer(attempts=100)
            add_mandarin_vocab(vocab, word, value)
            continue
        except SpaParseError:
            add_mandarin_vocab(vocab, word, w2v_dict[word])
print('vocab size: {}'.format(len(vocab.keys)))
print('oov number: {}'.format(len(oov)))
print('{}: {}'.format(oov[-1], vocab.pointers.get(oov[-1])))
