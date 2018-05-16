model_path = 'edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz'

# import os
# os.environ['STANFORD_PARSER'] = 'stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = 'stanford-parser-3.9.1-models.jar'
# s = u"媽媽 在擦洗 這個 東西 盤子 ， 這 想個 小朋友 在 旁邊 ， 站著 凳子 滿 危險的 ， 這個 小朋友 在 拿 點心 給 妹妹 ， 媽媽 在洗盤子 ， 水 滴 了 滿地 ， 水龍頭 開了 沒關 ， 而且 小朋友 這個 凳子 滿危險的 斜斜的 。 "
# s = u"媽媽 擦洗 東西 盤子 ， 小朋友 旁邊 ， 站著 凳子 危險的 ， 小朋友 拿 點心 給 妹妹 ， 媽媽 洗盤子 ， 水 滴  滿地 ， 水龍頭 開 沒關 ， 小朋友 凳子 危險的 斜斜的 。 "
s = "媽媽 擦洗 東西 盤子"

# 依存分析
from nltk.parse.stanford import StanfordDependencyParser
parser = StanfordDependencyParser(model_path=model_path)
result = list(parser.parse(s.split()))
for row in result[0].triples():
    print(row)

# 句法结构分析
from nltk.parse.stanford import StanfordParser
parser = StanfordParser(model_path=model_path)
result = list(parser.parse(s.split()))
for r in result:
    print(r)
    print(r.draw())
