import jieba.posseg as pseg
import jieba
import collections
import gensim
import numpy as np
WORDVEC_MODEL = '../wordvec_model/'
WV_DIIM = 500
def load_wordvec_model(file_name):
	w2v_model = gensim.models.Word2Vec.load(WORDVEC_MODEL+file_name)
	words = []
	for word in w2v_model.wv.vocab:
		words.append(word)
	print('Load word2vec model sucess ...')
	print('Number of token: {}'.format(len(words)))
	print('Dimensions of word vector: {}'.format(len(w2v_model[words[0]])))
	return w2v_model

def pos_tag_analysis(sentence, segment_tool): # segment_tool, 0:jieba, 1: Ckip
    if segment_tool=='jieba':
    	word_pos = pseg.cut(sentence)
    # elif segment_tool=='ckip':
    # 	segmenter = ckip.CKIP()
    # 	word_pos = segmenter.Segment(sentence)
    # 	word_pos = word_pos.res
    
    tmp_n, tmp_v, tmp_a, tmp_r, tmp_token = 0.0, 0.0, 0.0, 0.0, 0.0
    word_type = collections.Counter()
    for word, flag in word_pos:
        word_type[word] += 1
        tmp_token += 1
        if flag[0] == 'n':
            tmp_n += 1
        elif flag[0] == 'v':
            tmp_v += 1
        elif flag[0] == 'a':
            tmp_a += 1
        elif flag[0] == 'r':
            tmp_r += 1
    return [tmp_n/tmp_token, tmp_v/tmp_token, tmp_a/tmp_token, tmp_r/tmp_token, len(word_type)/tmp_token]

def semantic_analysis(sentence, w2v_model, zh_type='s'):

	vector = np.zeros((WV_DIIM))
	oov_num = 0
	# w2v_model = load_wordvec_model(WORDVEC_MODEL+'700features_20context_20mincount_zht')
	if zh_type=='s':
		# openCC = OpenCC('tw2s')
		# sentence = openCC.convert(sentence)
		token_sentence = jieba.lcut(sentence)
		# token_sentence = [t for t in token_sentence if not t in exclude]
	
	# elif zh_type=='tw':
	# 	segmenter = ckip.CkipSegmenter()
	# 	token_sentence = segmenter.seg(sentence)
	# 	token_sentence = [t for t in token_sentence.tok if not t in exclude]

	for token in token_sentence:
		if token in w2v_model.wv.vocab:
			vector += w2v_model[token]
		else:
			oov_num += 1
	vector /= len(token_sentence)
	return vector
