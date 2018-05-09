# import module
import data_preprocess
import nengo
from nengo.exceptions import SpaParseError, ValidationError
import sys
import numpy as np
from nengo.spa import pointer

DIM = 100
W2V_MODEL = '100features_20context_20mincount_zht'
w2v_model, _, w2v_dict = data_preprocess.load_wordvec_model(W2V_MODEL)
vocab = nengo.spa.Vocabulary(100, max_similarity=0.3) # optional max_similarity: 0.1

CONTROL_TRAIN = 'control.txt'
DEMENTIA_TRAIN = 'dementia.txt'
x_train, y_train = data_preprocess.read_sentence(DEMENTIA_TRAIN, CONTROL_TRAIN)
x_train_seg = data_preprocess.segmentation(x_train)

def add_mandarin_vocab(vocab, key, p):

    if not isinstance(p, pointer.SemanticPointer):
        p = pointer.SemanticPointer(p)
    if key in vocab.pointers:
        pass
        # print("The semantic pointer {} already exists".format(key))
    else:
        vocab.pointers[key] = p
        vocab.keys.append(key)
        vocab.vectors = np.vstack([vocab.vectors, p.v])
        # Generate vector pairs
        if vocab.include_pairs and len(vocab.keys) > 1:
            for k in vocab.keys[:-1]:
                vocab.key_pairs.append('%s*%s' % (k, key))
                v = (vocab.pointers[k] * p).v
                vocab.vector_pairs = np.vstack([vocab.vector_pairs, v])

oov = []
for sentence in x_train_seg:
    for word in sentence.split():
        try:
            vocab.add(word, w2v_dict[word])
        except KeyError:
            oov.append(word)
            value = vocab.create_pointer(attempts=100)
            add_mandarin_vocab(vocab, word, value)
            continue
        except SpaParseError:
            add_mandarin_vocab(vocab, word, w2v_dict[word])
print('vocab size: {}'.format(len(vocab.keys)))
print('oov number: {}'.format(len(oov)))
print('{}: {}'.format(oov[-1], vocab.pointers.get(oov[-1])))
