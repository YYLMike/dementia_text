# Dementia

file list: wikiXMLtoTXT.py, train_word2vec_model.py, data_preprocess.py, tokenize_data_helper.py,               syntactic_extract.py, semantic_extract.py, semantic_pointer_sentence_embedding.py,                   cluster.py, test_cluster.py, 
           CNN_text_classification.py

Section 1: Word Embedding

wikiXMLtoTXT.py:
    將wiki corpus的XML轉換成TXT檔，每一個換行符代表一篇文章

train_word2vec_model.py: 
    包含了兩個functions，一個是將corpus進行分詞並儲存，會用到dict.txt.big來擴充jieba分詞的辭典；一個是word embedding投影向量空間訓練

Section 2: Data Preprocess

data_preprocess.py: 
    將收集到的control資料由CSV轉成TXT檔;每一筆包含一行的ID碼與一行的敘述;每一比包含一句話與是否來自失智患者；將label從二為轉為一維;分詞;分詞加詞性標注;將段落依標點符號切分為句子；load word2vec model by gensim；產生sentence vector by average of word vectors

tokenize_data_helper.py:
    tokenize data by keras Tokenizer; pad tokenized list; inverse tokens to string;
    get embedding matrix using pretrained w2v

Section 3: Syntactic and Semantic Clustering

Syntacitc feature extract by using postag, semantic feature extract by sentence vector
Using K-mean cluster on control and dementia data
Plot scatter of the result
Predicting a unknown sentence is dementia or not by k-mean precit function

Section 4: CNN text mining, 
Using CNN on text raw data with three filter_size (3,4,5) and each has 128 filters to extract feature automaticly.
Classification layer is also connect on the last layer using softmax cross entropy.
Two types of embedding layer is used: one hot encoding and pretrained word2vec model.
Training step is using one-of-ten cross validation
Test step is using 5 of each class, i.e. 5 control& 5 dementia
