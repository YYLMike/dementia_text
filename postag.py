import jieba.posseg as pseg
import collections
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

CORPUS_PATH = "../data/"
with open(CORPUS_PATH+'dementia.txt', encoding='utf8') as f:
	sentence = f.readlines()
f.close()
sentence_list = [sentence[i] for i in range(len(sentence)) if i%2==1]

score = []
for s in sentence_list:
	word_pos = pseg.cut(s)
	tmp_n, tmp_v, tmp_a, tmp_r, tmp_token = 0.0, 0.0, 0.0, 0.0, 0.0
	word_type = collections.Counter() # for type token ratio, return a dict()
	for word, flag in word_pos:
		word_type[word] += 1
		tmp_token += 1
		if(flag[0] == 'n'):
			tmp_n += 1
		elif (flag[0] == 'v'):
			tmp_v += 1
		elif (flag[0] == 'a'):
			tmp_a += 1
		elif (flag[0] == 'r'):
			tmp_r += 1
	score.append([tmp_n/tmp_token, tmp_v/tmp_token, tmp_a/tmp_token, tmp_r/tmp_token, len(word_type)/tmp_token, tmp_token])
	# score.append([tmp_r/tmp_token])
# print(score)

# file = open("score.txt", 'w')
# for i in range(len(score)):
# 	for j in score[i]:
# 		file.write(str(j) + ' ')
# 	file.write('\n')
# file.close()
score = np.array(score)
kmeans = KMeans(n_clusters=2, random_state=1).fit(score)
print(kmeans.labels_)
### Reduce Dimension For Visualization
pca = PCA(n_components=2).fit(score)
score_2d = pca.transform(score)

### PLOTTING
plt.figure()
for i in range(0, score_2d.shape[0]):
    if(kmeans.labels_[i]==0):
        c1 = plt.scatter(score_2d[i,0],score_2d[i,1],c='r',marker='+')
        if(i==(len(score_2d)-1)):
            c3 = plt.scatter(score_2d[i,0],score_2d[i,1],c='g',marker='*')
    elif(kmeans.labels_[i]==1):
        c2 = plt.scatter(score_2d[i,0], score_2d[i,1],c='b', marker='.')
        if(i==(len(score_2d)-1)):
            c3 = plt.scatter(score_2d[i,0],score_2d[i,1],c='g',marker='*')
plt.legend([c1, c2], ['dementia', 'control'])
plt.title('Dementia')
plt.show()
