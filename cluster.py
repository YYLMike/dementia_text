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

# File path
SENTENCE_DICT = "../pickle/sentence_dict.pickle"
WORDVEC_MODEL = "../w2v_model/"
SENTENCE_WV = "../pickle/"

# Global variable
DEMENTIA_NUM = 51
CONTROL_NUM = 51
DATA_NUM = DEMENTIA_NUM + CONTROL_NUM
FEATURE_TYPE = ['Syntactic', 'Semantic', 'Syntactic_Semantic']


class Cluster:

    def __init__(self):
        self.sentence_dict = {}
        self.kmeans_cluster = []
        self.pca_labels = []
        # use syntactic_extract module
        self.postag_cls = syntactic_extract.Postag_analysis()
        self.syntactic_feature = []

        self.load_sentence_dict()
        self.sentence_vector_array = None

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

    def main():
    	pass
        # test_cluster.load_sentence_vector("s2v_array_zhs_500dim.pickle")
        # _, result_1_syntactic = test_cluster.evaluate('Syntactic')
        # _, result_1_semantic = test_cluster.evaluate('Semantic')
        # _, result_1_both = test_cluster.evaluate('Syntactic_Semantic')

        # print('Start Scenario 1, Kmean Clustering with semi-labeled data\nUsing jieba syntactic, jieba semantic features ...')
        # test_cluster = Cluster()
        # test_cluster.syntactic_extract('jieba')
        # test_cluster.load_sentence_vector("s2v_array_zhs_500dim.pickle")
        # _, result_1_syntactic = test_cluster.evaluate('Syntactic', 'result_1_syntactic.pickle')
        # _, result_1_semantic = test_cluster.evaluate('Semantic', 'result_1_semantic.pickle')
        # _, result_1_both = test_cluster.evaluate('Syntactic_Semantic', 'result_1_both.pickle')

        # print('Start Scenario 2, Kmean Clustering with semi-labeled data\nUsing ckip syntactic, ckip semantic features ...')
        # test_cluster2 = Cluster()
        # test_cluster2.syntactic_extract('ckip')
        # test_cluster2.load_sentence_vector("s2v_array_zht_500dim.pickle")
        # _, result_2_syntactic = test_cluster2.evaluate('Syntactic', 'result_2_syntactic.pickle')
        # _, result_2_semantic = test_cluster2.evaluate('Semantic', 'result_2_semantic.pickle')
        # _, result_2_both = test_cluster2.evaluate('Syntactic_Semantic', 'result_2_both.pickle')

        # print('Start Scenario 3, Kmean Clustering with semi-labeled data\nUsing jieba syntactic, ckip semantic features ...')
        # test_cluster3 = Cluster()
        # test_cluster3.syntactic_extract('jieba')
        # test_cluster3.load_sentence_vector('s2v_array_zht_500dim.pickle')
        # _, result_3_both = test_cluster3.evaluate('Syntactic_Semantic', 'result_3_both.pickle')

        # print('Start Scenario 4, Kmean Clustering with semi-labeled data\nUsing ckip syntactic, jieba semantic features ...')
        # test_cluster4 = Cluster()
        # test_cluster4.syntactic_extract('ckip')
        # test_cluster4.load_sentence_vector('s2v_array_zhs_500dim.pickle')
        # _, result_4_both = test_cluster4.evaluate('Syntactic_Semantic', 'result_4_both.pickle')


if __name__ == '__main__':
    pass