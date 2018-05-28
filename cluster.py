###############################################################################################################
#This program is a sklearn based kmean cluster model. Input file are dementia and control subjects' text file.#
#Output is a figure of PCA distribution from reducing dimension of KMeans clustering on syntactic features.####
# This script is written in Python 3.6
###############################################################################################################

# Import modules
# Basic tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle

# Machine learning Tools
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Custumer module for feature extraction
import syntactic_extract
import semantic_extract
# File path
SENTENCE_DICT = "../pickle/sentence_dict.pickle"
WORDVEC_MODEL = "../w2v_model/"
MODEL_NAME = "500features_20context_20mincount"
SENTENCE_WV = "../pickle/"

# Global variable
DEMENTIA_NUM = 51
CONTROL_NUM = 51
DATA_NUM = DEMENTIA_NUM + CONTROL_NUM
FEATURE_TYPE = ['Syntactic', 'Semantic', 'Syntactic_Semantic']
CKIP_POS = '../pickle/ckip_AD_control_pos_score.pickle'
CKIP_SENTENCE_SEG = '../pickle/ckip_AD_control_seg.pickle'

class Cluster:

    def __init__(self, model_name=MODEL_NAME):
        self.sentence_dict = {}
        self.kmeans_cluster = []
        self.pca_labels = []
        # use syntactic_extract module
        self.postag_cls = syntactic_extract.Postag_analysis()
        self.syntactic_feature = []
        # use semantic_extract module
        self.paragraph_vec = semantic_extract.Semantic_analysis(model_name)
        self.semantic_feature = []
        # load the textual data
        self.load_sentence_dict()

    def load_sentence_dict(self):
        with open(SENTENCE_DICT, 'rb') as f:
            self.sentence_dict = pickle.load(f)
        print("Load sentence text data ...")

    def syntactic_analysis(self, segment_tool='jieba'):
        for key, s in self.sentence_dict.items():
            self.postag_cls.pos_tag_analysis(s, segment_tool)
        if segment_tool=='jieba':
            self.syntactic_feature = self.postag_cls.syntactic_features_jieba.copy()
        elif segment_tool=='ckip':
            self.syntactic_feature = self.postag_cls.syntactic_features_ckip.copy()
        print('Syntactic features has extracted ...')
    
    # save time for analysis, since ckip online cost times
    def syntactic_ckip_load(self):
        with open(CKIP_POS, 'rb') as f:
            self.syntactic_feature = pickle.load(f)
        print('Syntactic features using ckip has extracted ...')

    def write_syntactic_feature(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.syntactic_feature, f)
        print('Write syntactic features success...')

    def semantic_analysis(self, segment_tool='jieba'):
        if segment_tool=='jieba':
            for key, s in self.sentence_dict.items():
                self.paragraph_vec.generate_sentence_vec_avg(s, 's')
            self.semantic_feature = self.paragraph_vec.paragraph_vec.copy()
        # save time for analysis, since ckip online cost times, using pre-segment ckip sentences
        elif segment_tool=='ckip':
            with open(CKIP_SENTENCE_SEG, 'rb') as f:
                sentences = pickle.load(f)
            for i in sentences:
                self.paragraph_vec.generate_sv_avg_ckip(i)
            self.semantic_feature = self.paragraph_vec.paragraph_vec.copy()
        print('Semantic features has extracted ...')

    def k_mean_cluster(self, feature_mode='Syntactic'): # 0: syntactuc, 1: semantic, 2: both
        if feature_mode==FEATURE_TYPE[0]:
            score = np.array(self.syntactic_feature)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(score)
            pca = PCA(n_components=2).fit(score)
            score_2d = pca.transform(score)

        elif feature_mode==FEATURE_TYPE[1]:
            score = self.semantic_feature
            kmeans = KMeans(n_clusters=2, random_state=0).fit(score)
            pca = PCA(n_components=2).fit(score)
            score_2d = pca.transform(score)
        
        elif feature_mode==FEATURE_TYPE[2]:
            syntactic_f = np.array(self.syntactic_feature)
            semantic_f = self.semantic_feature
            score = np.concatenate((syntactic_f, semantic_f), axis=1)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(score)
            pca = PCA(n_components=2).fit(score)
            score_2d = pca.transform(score)

        print('Kmean and PCA success...')
        return kmeans, score_2d

    def evaluate(self, feature_mode='Syntactic', result_name=None):
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
        print('False Positive: {}'.format(fp))
        print('False Negative: {}'.format(fn))
        accuracy = (len(tp) + len(tn)) / DATA_NUM
        precision = len(tp)/(len(tp)+len(fp))
        recall = len(tp)/(len(tp)+len(fn))
        print("Accuracy :{0}\nPrecision: {1}\nRecall: {2}\n".format(accuracy, precision, recall))

        # dump cluster result into pickle files
        result_dict = {}
        result_dict['tp'] = tp
        result_dict['fp'] = fp
        result_dict['tn'] = tn
        result_dict['fn'] = fn

        if result_name:
            with open(result_name, 'wb') as f:
                pickle.dump(result_dict, f)

        return accuracy, result_dict

    def plot_cluster(self, title_name=None):
        plt.figure()
        # kmeans, score_2d = self.k_mean_cluster()
        for i in range(0, self.pca_labels.shape[0]):
            if self.kmeans_cluster.labels_[i]==0:
                c1 = plt.scatter(self.pca_labels[i, 0], self.pca_labels[i, 1], c='r', marker='+')
            if self.kmeans_cluster.labels_[i]==1:
                c2 = plt.scatter(self.pca_labels[i, 0], self.pca_labels[i, 1], c='b', marker='.')
        plt.legend([c1, c2], ['dementia', 'control'])
        plt.title('K mean cluster'+title_name)
        plt.savefig(title_name+'.png')

    def predict_sentence(self, sentence):
        score = np.array(self.pos_tag_analysis(sentence))
        score = np.expand_dims(score, axis=0) ## predict needs at least 2 dim
        print(self.kmeans_cluster.predict(score))
