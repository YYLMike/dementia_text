sentence_num = []
with open('sentence_num.txt', 'r') as f:
    for i in f.readlines():
        sentence_num.append(i)
sentence_depth = []
with open('sentence_depth.txt', 'r') as f:
    for i in f.readlines():
        sentence_depth.append(i)

avg_sent_depth = []

idx = 0
for i in range(len(sentence_num)):
    tmp_sum = 0
    for j in range(int(sentence_num[i])):
        tmp_sum += sentence_depth[j+idx]
    avg_sent_depth.append(tmp_sum/sentence_num[i])
    idx += sentence_num[i]
with open('avg_sent_depth.txt', 'w') as f:
    for i in avg_sent_depth:
        f.write(str(i))
        f.write('\n')
