# import modules
import pandas as pd
import pickle
import gensim
import numpy as np
import string
from opencc import OpenCC
import ckip
# Path of files
SENTENCE_DICT = "../pickle/sentence_dict.pickle"
WORDVEC_MODEL = '../wordvec_model/'
# Variables
DEMENTIA_NUM = 51
CONTROL_NUM = 51
WV_DIIM = 700

def read_sentence_file(file_name=None):
	with open(SENTENCE_DICT, 'rb') as f:
    	self.sentence_dict = pickle.load(f)
    	print("Load sentence text data ...")

def load_wordvec_model(file_name):
	w2v_model = gensim.models.Word2Vec.load(WORDVEC_MODEL+file_name)
	words = []
	for word in w2v_model.wv.vocab:
		words.append(word)
	print('Load word2vec model sucess ...')
	print('Number of token: {}'.format(len(words)))
	print('Dimensions of word vector: {}'.format(len(w2v_model[words[0]])))
	return w2v_model

def generate_sentence_vec_avg(sentence, zh_type, w2v_model):
	exclude = set(string.punctuation + '，'+'。'+'、'+'「'+'」'+'？'+'！')
	vector = np.zeros((WV_DIIM))
	oov_num = 0
	
	if zh_type=='s':
		openCC = OpenCC('tw2s')
		sentence = openCC.convert(sentence)
		token_sentence = jieba.lcut(sentence)
		token_sentence = [t for t in token_sentence if not t in exclude]
	
	elif zh_type=='tw':
		segmenter = ckip.CkipSegmenter()
		token_sentence = segmenter.seg(sentence)
		token_sentence = [t for t in token_sentence.tok if not t in exclude]

	for token in token_sentence:
		if token in w2v_model.wv.vocab:
			vector += w2v_model[token]
		else:
			oov_num += 1
	vector /= len(token_sentence)
	return vector, oov_num, token_sentence

def write_sentence_vec_avg_dict(sentence_dict, file_name=None):
	sv_dict = {}
	oov_dict = {}
	sentence_token_dict = {}
	w2v_model = load_wordvec_model(WORDVEC_MODEL+'500features_20context_20mincount_zht')
	for key, sentence in sentence_dict.items():
		sv, oov, sentence_token = generate_sentence_vec_avg(sentence, 'tw', w2v_model)
		sv_dict[key] = sv
		oov_dict[key] = oov
		sentence_token_dict[key] = sentence_token
	sv_dict_array = np.asarray([i for i in sv_dict.values()])
	print('sentence vector array load sucess ...')

	if file_name:
		with open('s2v_array_'+str(file_name) + '_' + WV_DIIM + 'dim.pickle', 'wb') as f:
			pickle.dump(sv_dict_array, f)
			print('Write sentence vector array to pickle ...')
	return sv_dict_array

if __name__ == '__main__':

	csv_to_txt('control.txt')

	sentence_dict = read_sentence_file()

	# sv_dict_array = write_sentence_vec_avg_dict(sentence_dict, '_zht_')
