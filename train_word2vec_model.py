#!/usr/bin/env python  
# -*- coding: utf-8 -*-
##########################
# This script is a word2vec training program. The output is a model for gensim to call.
# ########################
import logging   
import sys  
import multiprocessing  
   
from gensim.models import word2vec  
from gensim.models.word2vec import LineSentence  
import multiprocessing
import re
import jieba

JIEBA_DICT = '../data/dict.txt.big'
SEG_CORPUS = '../data/'
jieba.set_dictionary(JIEBA_DICT)
# Segment the corpus text file, this will take a while
def corpus_to_segment_list(corpus):
    file_write = open(SEG_CORPUS+'seg_wiki.zht.txt', 'w')
    with open(corpus) as f:
        for line in f:
            # Delete all words not chinese
            line = re.sub('[^\u4e00-\u9fff]', '', line)
            if len(line) > 2:
                seg_line = jieba.lcut(line)
                for token in seg_line:
                    file2.write(token + ' ')
                file2.write('\n')
        file2.close()

def train_w2v_model(file_name):
    num_features = 500
    context = 20
    min_count = 20
    workers = multiprocessing.cpu_count()
    model = word2vec.Word2Vec(LineSentence(file_name), size=num_features, 
                    window=context, min_count=min_count, workers=workers)
    return model
if __name__ == '__main__':
    
    # seg_wiki_zhs_list = corpus_to_segment_list('../data/wiki.zht.txt')

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')  
    logging.root.setLevel(level=logging.INFO)  
    model = train_w2v_model(SEG_CORPUS+'seg_wiki.zht.txt')
    model.init_sims(replace=True)
    model.save(str(num_features)+'_features_'+str(context)+'_context_'+str(min_count)+'_mincount_zht')
    # model = FastText(LineSentence(inp), size=100, window=5, min_count=5,
    #         workers=multiprocessing.cpu_count())
