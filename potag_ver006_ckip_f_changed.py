###############################################################################################################
#This program is a sklearn based kmean cluster model. Input file are dementia and control subjects' text file.#
#Output is a figure of PCA distribution from reducing dimension of KMeans clustering on syntactic features.####
#This script is written in Python 3.6
###############################################################################################################

# Import modules
# Basic tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle
# Natural language process tools
import jieba.posseg as pseg
import ckip
import collections
from opencc import OpenCC
# Machine learning Tools
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# File path
SENTENCE_DICT = "../pickle/sentence_dict.pickle"
WORDVEC_MODEL ="../w2v_model/"
SENTENCE_WV = "../pickle/"

# Global variable
DEMENTIA_NUM = 51
CONTROL_NUM = 51
DATA_NUM = DEMENTIA_NUM + CONTROL_NUM
FEATURE_TYPE = ['Syntactic', 'Semantic', 'Syntactic_Semantic']

# Pos-tag type for Ckip Segmenter

segmenter = ckip.CkipSegmenter()

class Cluster:


    def __init__(self):
        self.sentence_dict = {}
        self.sentence_vector_array = None
        self.syntactic_feature = []
        self.kmeans_cluster = []
        self.pca_labels = []

        self.load_sentence_dict()

    def load_sentence_dict(self):
    	with open(SENTENCE_DICT, 'rb') as f:
    		self.sentence_dict = pickle.load(f)
    	print("Load sentence text data ...")

    def pos_tag_analysis(self, sentence, segment_tool): # segment_tool, 0:jieba, 1: Ckip
        if segment_tool=='jieba':
        	openCC = OpenCC('tw2s')
        	sentence = openCC.convert(sentence) 
        	word_pos = pseg.cut(sentence)
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

    def syntactic_extract(self, segment_tool='jieba'):
    	self.syntactic_feature = []
    	
    	if segment_tool=='jieba':
	    	for key, s in self.sentence_dict.items():
	    		pos_tag_result = self.pos_tag_analysis(s, segment_tool)
	    		self.syntactic_feature.append(pos_tag_result)
    	elif segment_tool=='ckip':
            with open('../pickle/ckip_seg_result.pickle', 'rb') as f:
                ck_result = pickle.load(f)
            for word_pos in ck_result:
                tmp_na, tmp_d, tmp_va, tmp_vk, tmp_vh, tmp_vcl, tmp_dfa, tmp_nh, tmp_cbb, tmp_vj, tmp_ng, tmp_v2, tmp_token = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                word_type = collections.Counter()
                for word, flag in word_pos:
                    word_type[word] += 1
                    tmp_token += 1
                    if flag == 'Na':
                        tmp_na += 1
                    elif flag =='D':
                        tmp_d += 1
                    elif flag == 'VA':
                        tmp_va += 1
                    elif flag == 'VK':
                        tmp_vk += 1
                    elif flag == 'VH':
                        tmp_vh += 1
                    elif flag == 'VCL':
                        tmp_vcl += 1
                    elif flag == 'Dfa':
                        tmp_dfa += 1
                    elif flag == 'Nh':
                        tmp_nh += 1
                    elif flag == 'Cbb':
                        tmp_cbb += 1
                    elif flag == 'VJ':
                        tmp_vj += 1
                    elif flag == 'Ng':
                        tmp_ng += 1
                    elif flag == 'V_2':
                        tmp_v2 += 1
                self.syntactic_feature.append([tmp_na/tmp_token, tmp_d/tmp_token, tmp_va/tmp_token, tmp_vk/tmp_token, 
                         tmp_vh/tmp_token, tmp_vcl/tmp_token, tmp_dfa/tmp_token, tmp_nh/tmp_token, 
                         tmp_cbb/tmp_token, tmp_vj/tmp_token, tmp_ng/tmp_token, tmp_v2/tmp_token])
    	print('Syntactic features extract success...')

    def write_syntactic_feature(self, file_name):
        file = open(file_name, "w", encoding='utf8')
        for i in range(len(self.syntactic_feature)):
            for j in self.syntactic_feature[i]:
                file.write(str(j) + ' ')
            file.write('\n')
        file.close()
        print('Write syntactic features success...')

    # New Feature
    def load_sentence_vector(self, model):
    	with open(SENTENCE_WV+str(model), 'rb') as f:
    		self.sentence_vector_array = pickle.load(f)

    def k_mean_cluster(self, feature_mode='Syntactic'): # 0: syntactuc, 1: semantic, 2: both
        if feature_mode==FEATURE_TYPE[0]:
        	score = np.array(self.syntactic_feature)
        	kmeans = KMeans(n_clusters=2, random_state=0).fit(score)
        	pca = PCA(n_components=2).fit(score)
        	score_2d = pca.transform(score)

        elif feature_mode==FEATURE_TYPE[1]:
        	score = self.sentence_vector_array
        	kmeans = KMeans(n_clusters=2, random_state=0).fit(score)
        	pca = PCA(n_components=2).fit(score)
        	score_2d = pca.transform(score)
        
        elif feature_mode==FEATURE_TYPE[2]:
        	syntactic_f = np.array(self.syntactic_feature)
        	semantic_f = self.sentence_vector_array
        	score = np.concatenate((syntactic_f, semantic_f), axis=1)
        	kmeans = KMeans(n_clusters=2, random_state=0).fit(score)
        	pca = PCA(n_components=2).fit(score)
        	score_2d = pca.transform(score)

        print('Kmean and PCA success...')
       	return kmeans, score_2d

    def evaluate(self, feature_mode='Syntactic'):
        self.kmeans_cluster, self.pca_labels = self.k_mean_cluster(feature_mode)
        dementia = self.kmeans_cluster.labels_[:DEMENTIA_NUM]
        control = self.kmeans_cluster.labels_[DEMENTIA_NUM:]

        counts = np.bincount(control)
        control_target = np.argmax(counts) # Mode

        tp = np.asarray(np.where(control==control_target)).flatten()
        tn = np.asarray(np.where(dementia!=control_target)).flatten()
        fp = np.asarray(np.where(dementia==control_target)).flatten()
        fn = np.asarray(np.where(control!=control_target)).flatten()
        print(str(feature_mode) + " features evaluation results.")
        print("True Positve: {0}\nFalse Positve: {1}\nTrue Negative: {2}\nFalse Negative: {3}\n"\
        	.format(len(tp), len(fp), len(tn), len(fn)))
        accuracy = (len(tp) + len(tn)) / DATA_NUM
        precision = len(tp)/(len(tp)+len(fp))
        recall = len(tp)/(len(tp)+len(fn))
        print("Accuracy :{0}\nPrecision: {1}\nRecall: {2}\n".format(accuracy, precision, recall))
        return accuracy

    def plot_cluster(self):
        plt.figure()
        # kmeans, score_2d = self.k_mean_cluster()
        for i in range(0, self.pca_labels.shape[0]):
            if self.kmeans_cluster.labels_[i]==0:
                c1 = plt.scatter(self.pca_labels[i, 0], self.pca_labels[i, 1], c='r', marker='+')
            if self.kmeans_cluster.labels_[i]==1:
                c2 = plt.scatter(self.pca_labels[i, 0], self.pca_labels[i, 1], c='b', marker='.')
        plt.legend([c1, c2], ['dementia', 'control'])
        plt.title('K mean cluster')
        plt.show()

    def predict_sentence(self, sentence):
        score = np.array(self.pos_tag_analysis(sentence))
        score = np.expand_dims(score, axis=0) ## predict needs at least 2 dim
        print(self.kmeans_cluster.predict(score))


if __name__ == '__main__':
    
    test_cluster = Cluster()
    
    print('Start Scenario 1, Kmean Clustering with semi-labeled data\nUsing jieba syntactic, jieba semantic features ...')

    test_cluster.syntactic_extract('jieba')
    test_cluster.load_sentence_vector("s2v_array_zhs_500dim.pickle")
    test_cluster.evaluate('Syntactic')
    test_cluster.evaluate('Semantic')
    test_cluster.evaluate('Syntactic_Semantic')

    print('Start Scenario 2, Kmean Clustering with semi-labeled data\nUsing ckip syntactic, ckip semantic features ...')
    test_cluster2 = Cluster()
    test_cluster2.syntactic_extract('ckip')
    test_cluster2.load_sentence_vector("s2v_array_zht_500dim.pickle")
    test_cluster2.evaluate('Syntactic')
    test_cluster2.evaluate('Semantic')
    test_cluster2.evaluate('Syntactic_Semantic')

    print('Start Scenario 3, Kmean Clustering with semi-labeled data\nUsing jieba syntactic, ckip semantic features ...')
    test_cluster3 = Cluster()
    test_cluster3.syntactic_extract('jieba')
    test_cluster3.load_sentence_vector('s2v_array_zht_500dim.pickle')
    test_cluster3.evaluate('Syntactic_Semantic')

    print('Start Scenario 4, Kmean Clustering with semi-labeled data\nUsing ckip syntactic, jieba semantic features ...')
    test_cluster4 = Cluster()
    test_cluster4.syntactic_extract('ckip')
    test_cluster4.load_sentence_vector('s2v_array_zhs_500dim.pickle')
    test_cluster4.evaluate('Syntactic_Semantic')

    # predict_sentence = "媽媽在洗碗時候洗手台的水流出來了，有兩個小孩在櫥櫃旁邊，男孩正站在椅子上要去拿餅乾，女孩在椅子旁伸手跟男孩要餅乾。"
    # test_cluster.predict_sentence(predict_sentence)
