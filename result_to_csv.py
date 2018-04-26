import postag_ver006
import numpy as np
import pandas as pd

test_cluster = postag_ver006.Cluster()
test_cluster.syntactic_extract('jieba')
syntactic_feature_np = np.array(test_cluster.syntactic_feature).reshape(-1,5)
df = pd.DataFrame(syntactic_feature_np)
df.to_csv('jieba_synt.csv')
# print(syntactic_feature_np)
for i in range(5):
	print(np.mean(syntactic_feature_np[:51][i]))
	print(np.mean(syntactic_feature_np[51:][i]))

test_cluster_2 = postag_ver006.Cluster()
test_cluster_2.syntactic_extract('ckip')
syntactic_feature_np_ck = np.array(test_cluster_2.syntactic_feature).reshape(-1,5)
df = pd.DataFrame(syntactic_feature_np_ck)
df.to_csv('ckip_synt.csv')
for i in range(5):
	print(np.mean(syntactic_feature_np_ck[:51][i]))
	print(np.mean(syntactic_feature_np_ck[51:][i]))
