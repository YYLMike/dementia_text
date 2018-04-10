# Dementia

git push https://github.com/YYLMike/Dementia.git
User name: YYLMike
User pwd: lkklkk123

file list: wikiXMLtoTXT.py, train_word2vec_model.py, data_preprocess.py, postag_ver*.py

Section 1: Word Embedding

wikiXMLtoTXT.py: 將wiki corpus的XML轉換成TXT檔，每一個換行符代表一篇文章
train_word2vec_model.py: 包含了兩個functions，一個是將corpus進行分詞並儲存，會用到dict.txt.big來擴充jieba分詞的辭典；一個是word embedding投影向量空間訓練

Section 2: Data Preprocess

data_preprocess.py: 將收集到的control資料由CSV轉成TXT檔，每一筆包含一行的ID碼與一行的敘述；將dementia, control的敘述存到一個dictionary；load word2vec model by gensim；產生sentence vector by average of word vectors；儲存sentence vector至dictionary

Section 3: Syntactic and Semantic Clustering

Syntacitc feature extract by using postag, semantic feature extract by sentence vector
Using K-mean cluster on control and dementia data
Plot scatter of the result
Predicting a unknown sentence is dementia or not by k-mean precit function