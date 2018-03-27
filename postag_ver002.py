import jieba.posseg as pseg
import collections
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

CORPUS_PATH = "../data/"
SAVE_SCORE = False
sentence_dict = {} # key=id, value=sentence
with open(CORPUS_PATH+'dementia.txt', encoding='utf8') as f:
	sentence = f.readlines()
f.close()
# sentence_list = [sentence[i] for i in range(len(sentence)) if i%2==1]
for i in range(len(sentence)):
	if i%2==0:
		sentence_dict[sentence[i].strip('\n')] = sentence[i+1].strip('\n')
with open(CORPUS_PATH+'control_42.txt', encoding='utf8') as f:
	sentence = f.readlines()
f.close()
for i in range(len(sentence)):
	if i%2==0:
		sentence_dict[sentence[i].strip('\n')] = sentence[i+1].strip('\n')
# print(sentence_dict)

score = []
for key,s in sentence_dict.items():
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
	score.append([tmp_n/tmp_token, tmp_v/tmp_token, tmp_a/tmp_token, tmp_r/tmp_token, len(word_type)/tmp_token])
	# score.append([tmp_r/tmp_token, len(word_type)/tmp_token])
print(score)

if SAVE_SCORE==True:
	file = open("score_ver002.txt", 'w')
	for i in range(len(score)):
		for j in score[i]:
			file.write(str(j) + ' ')
		file.write('\n')
	file.close()
score = np.array(score)
kmeans = KMeans(n_clusters=2, random_state=0).fit(score)
print(kmeans.labels_)
## Reduce Dimension For Visualization
pca = PCA(n_components=2).fit(score)
score_2d = pca.transform(score)

dementia = kmeans.labels_[:52]
control = kmeans.labels_[52:]
print(control)
wrong = 0
for i in control:
	if(i==0):
		wrong += 1
for i in dementia:
	if(i==1):
		wrong += 1
accuracy = 1-(wrong/len(kmeans.labels_))
print('accuracy: '+ str(accuracy))
### PLOTTING
plt.figure()
## Using Two Features
# for i in range(0, score.shape[0]):
# 	if kmeans.labels_[i]==0:
# 		c1 = plt.scatter(score[i,0],score[i,1],c='r',marker='+')
# 	elif kmeans.labels_[i]==1:
# 		c2 = plt.scatter(score[i,0], score[i,1],c='b', marker='.')

## Using more than two features
for i in range(0, score_2d.shape[0]):
	if kmeans.labels_[i]==0:
		c1 = plt.scatter(score_2d[i,0],score_2d[i,1],c='r',marker='+')
	elif kmeans.labels_[i]==1:
		c2 = plt.scatter(score_2d[i,0], score_2d[i,1],c='b', marker='.')
	else:
		c3 = plt.scatter(score_2d[i,0], score_2d[i,1],c='g', marker='o')

plt.legend([c1, c2], ['dementia', 'control'])
plt.title('Dementia')
plt.show()
