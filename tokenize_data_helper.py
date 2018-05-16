from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np

class tokenize_data_helper():

    def __init__(self, fit_data, num_words):
        self.tokenizer = self.tokenize(fit_data, num_words)
        self.idx = self.tokenizer.word_index
        self.inverse_map = dict(zip(self.idx.values(), self.idx.keys()))
        self.num_words = num_words

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
    
    def embedding_matrix(self, pretrained_dict, EMBEDDING_DIM):
        word_index = self.tokenizer.word_index
        word_embedding = np.zeros((self.num_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = pretrained_dict.get(word)
            if embedding_vector is not None:
                word_embedding[i] = embedding_vector
        return word_embedding

if __name__ == '__main__':
    pass