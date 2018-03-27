#!/usr/bin/env python  
# -*- coding: utf-8 -*-
#python train_word2vec_model.py tw_seg_wiki.zh.txt tw_wiki.zh.text.model tw_wiki.zh.vec
   
import logging  
import os.path  
import sys  
import multiprocessing  
   
from gensim.corpora import WikiCorpus  
from gensim.models import Word2Vec  
from gensim.models.word2vec import LineSentence  
from gensim.models.fasttext import FastText
if __name__ == '__main__':  
    program = os.path.basename(sys.argv[0])  
    logger = logging.getLogger(program)  
   
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')  
    logging.root.setLevel(level=logging.INFO)  
    logger.info("running %s" % ' '.join(sys.argv))  
   
    # check and process input arguments  
    if len(sys.argv) < 4:
        print (globals()['__doc__'] % locals())  
        sys.exit(1)  
    inp, outp1, outp2 = sys.argv[1:4]

    ###choose the word to vector training method
    model = Word2Vec(LineSentence(inp), size=1000, window=5, min_count=1,
            workers=multiprocessing.cpu_count(), hs=1, negative=0)
    # model = FastText(LineSentence(inp), size=100, window=5, min_count=5,
    #         workers=multiprocessing.cpu_count())
    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)