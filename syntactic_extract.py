import jieba.posseg as pseg
import ckip
import collections
from opencc import OpenCC

# Pos-tag type for Ckip Segmenter
noun_set = ('Na', 'Nb', 'Nc', 'Ncd', 'Nd', 'Neu',
            'Neqa', 'Neqb', 'Nf', 'Ng', 'Nv', 'n')
pronoun_set = ('Nh', 'Nep', 'r')
verb_set = ('VA', 'VAC', 'VB', 'VC', 'VCL', 'VD', 'VE', 'VF',
            'VG', 'VH', 'VHC', 'VI', 'VJ', 'VK', 'VL', 'V_2', 'v')
a_set = ('A', 'a')

class Postag_analysis:

    def __init__(self):

        self.segmenter = ckip.CkipSegmenter()
        self.syntactic_features_jieba = []
        self.syntactic_features_ckip = []

    def pos_tag_analysis(self, sentence, segment_tool):  # segment_tool, 0:jieba, 1: Ckip

        tmp_n, tmp_v, tmp_a, tmp_r, tmp_token = 0.0, 0.0, 0.0, 0.0, 0.0
        word_type = collections.Counter()

        if segment_tool == 'jieba':
            openCC = OpenCC('tw2s')
            sentence_s = openCC.convert(sentence)
            word_pos = pseg.cut(sentence_s)

        elif segment_tool == 'ckip':
            word_pos = self.segmenter.seg(sentence)
            word_pos = word_pos.res

        for word, flag in word_pos:
            word_type[word] += 1
            # print(word+flag, end=', ')
            tmp_token += 1
            if flag in noun_set:
                tmp_n += 1
            elif flag in verb_set:
                tmp_v += 1
            elif flag in a_set:
                tmp_a += 1
            elif flag in pronoun_set:
                tmp_r += 1
        if segment_tool == 'jieba':
            self.syntactic_features_jieba.append([tmp_n/tmp_token, tmp_v/tmp_token, tmp_a/tmp_token, tmp_r/tmp_token, len(word_type)/tmp_token])
        elif segment_tool == 'ckip':
            self.syntactic_features_ckip.append([tmp_n/tmp_token, tmp_v/tmp_token, tmp_a/tmp_token, tmp_r/tmp_token, len(word_type)/tmp_token])
        
        return [tmp_n/tmp_token, tmp_v/tmp_token, tmp_a/tmp_token, tmp_r/tmp_token, len(word_type)/tmp_token]

    def main():
        test_postag_analysis = Postag_analysis()

        test_sentence = '這個孩子拿了個東西放到上面，有一個流水台，這是他的家庭主婦，這是他的兒子。'
        postag_jieba = test_postag_analysis.pos_tag_analysis(
            test_sentence, 'jieba')
        print('*'*100)
        postag_ckip = test_postag_analysis.pos_tag_analysis(
            test_sentence, 'ckip')
        print(postag_jieba)
        print(postag_ckip)

        test_sentence = '這個小姐在洗碗，手裡拿了個盤子，這個小孩在偷點心吃，這個凳子快要翻了，這個小女孩在指責他 ，這個洗碗的盆子的水已經滿了都流出來了，這個窗外面有樹木，這也是窗外面的景色，這是一個櫥櫃。'
        postag_jieba = test_postag_analysis.pos_tag_analysis(
            test_sentence, 'jieba')
        print('*'*100)
        postag_ckip = test_postag_analysis.pos_tag_analysis(
            test_sentence, 'ckip')
        print(postag_jieba)
        print(postag_ckip)
        print('*'*100)
        print(test_postag_analysis.syntactic_features)

if __name__ == '__main__':
    Postag_analysis.main()
