###########
#This program is a sklearn based kmean cluster model. Input file are dementia and control subjects' text file.
#Output is a figure of PCA distribution from reducing dimension of KMeans clustering on syntactic features.
###########
import jieba.posseg as pseg 

import collections
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


DEMENTIA_PATH = "../data/dementia.txt"
CONTROL_PATH = "../data/control_51.txt"
CSV_PATH = "../data/CookieTheft_51.csv"
DEMENTIA_NUM = 51
CONTROL_NUM = 51


class Cluster:

    def __init__(self):
        self.sentence_dict = {}
        self.syntactic_feature = []
        self.kmeans_cluster = []
        self.pca_labels = []

        self.read_sentence_file()
        self.syntactic_extract()


    def csv_to_txt(self):
        csv = pd.read_csv(CSV_PATH, header=None)
        with open("control_42.txt", 'w', encoding='utf8') as f:
            idx = 68
            for line in csv.iloc[1:, 1]:
                f.write(str(idx) + '\n')
                f.write(line + '\n')
                idx += 1
        print('CSV to txt file success...')

    def read_sentence_file(self):
        with open(DEMENTIA_PATH, encoding='utf8') as f:
            sentence = f.readlines()
        f.close()
        for i in range(len(sentence)):
            if i % 2 == 0:
                self.sentence_dict[sentence[i].strip('\n')] = sentence[i+1].strip('\n')
        with open(CONTROL_PATH, encoding='utf8') as f:
            sentence = f.readlines()
        f.close()
        for i in range(len(sentence)):
            if i % 2 == 0:
                self.sentence_dict[sentence[i].strip('\n')] = sentence[i+1].strip('\n')
        print('Read file success...')

    def pos_tag_analysis(self, sentence):
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

    def syntactic_extract(self):
        for key, s in self.sentence_dict.items():
            pos_tag_result = self.pos_tag_analysis(s)
            self.syntactic_feature.append(pos_tag_result)
        print('Syntactic features extract success...')

    def write_syntactic_feature(self):
        file = open("score_ver003.txt", "w", encoding='utf8')
        file = open(sys.argv[1], "w", encoding='utf8')
        for i in range(len(self.syntactic_feature)):
            for j in self.syntactic_feature[i]:
                file.write(str(j) + ' ')
            file.write('\n')
        file.close()
        print('Write syntactic features success...')

    def k_mean_cluster(self):
        score = np.array(self.syntactic_feature)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(score)
        pca = PCA(n_components=2).fit(score)
        score_2d = pca.transform(score)
        print('Kmean and PCA success...')
        return kmeans, score_2d

    def evaluate(self):
        self.kmeans_cluster, self.pca_labels = self.k_mean_cluster()
        dementia = self.kmeans_cluster.labels_[:DEMENTIA_NUM]
        control = self.kmeans_cluster.labels_[DEMENTIA_NUM:]
        # print("dementia: \n" + str(dementia))
        print("False Positive: " + str(np.where(dementia == 1)))
        # print("control: \n" + str(control))
        print("False Negative: " + str(np.where(control == 0)))
        wrong = 0
        for i in control:
            if i == 0:
                wrong += 1
        for i in dementia:
            if i == 1:
                wrong += 1
        accuracy = 1 - (wrong/len(self.kmeans_cluster.labels_))
        print('Evaluate success...\n accuracy: ', str(accuracy))
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
    print('Start Scenario 1, Kmean Clustering with semi-labeled data...')
    test_cluster = Cluster()
    # test_cluster.read_sentence_file()
    # test_cluster.syntactic_extract()
    # test_cluster.csv_to_txt()
    # test_cluster.write_syntactic_feature()
    if len(sys.argv) < 2:
        print('Do not write score file...')
    else:
        test_cluster.write_syntactic_feature()
    test_cluster.evaluate()
    test_cluster.plot_cluster()
    predict_sentence = "媽媽在洗碗時候洗手台的水流出來了，有兩個小孩在櫥櫃旁邊，男孩正站在椅子上要去拿餅乾，女孩在椅子旁伸手跟男孩要餅乾。"
    test_cluster.predict_sentence(predict_sentence)
