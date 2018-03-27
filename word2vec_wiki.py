import gensim

import logging

CORPUS_PATH = "training_data/"
PICKLE_PATH = "pickle/"
model = gensim.models.Word2Vec.load("tw_wiki.text.model")

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


for i in range(1,2):
    file = open(CORPUS_PATH + "cut_"+str(i)+"_train.txt", encoding='utf-8')
    sentences = file.readlines()
    file.close()
    # tmp_file = []
    OOV = set([])
    file = open(PICKLE_PATH + "wordVector_cut_" + str(i) + ".txt", "w")
    for line in sentences:
        # tmp_line = []
        for token in line.split():
            if token not in model.wv.vocab:
                # logging.debug("%s not in vocab", token)
                OOV.add(token)
                continue
            else:
                vector = model.wv[token]
                for v in vector:
                    file.write(str(v)+' ')
                # tmp_line.append(vector)
        # tmp_file.append(tmp_line)
        file.write('\n')
    file.close()
    write_file = open(PICKLE_PATH + "outOfVocab_words_"+str(i)+".txt", 'w')
    for j in OOV:
        write_file.write(j+'\n')
    write_file.close()




