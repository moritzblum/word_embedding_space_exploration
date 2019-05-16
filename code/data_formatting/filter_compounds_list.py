import json

import os
from gensim.models import KeyedVectors

MODEL_PATH = '../data/embedding_models/'
word2vec = 'GoogleNews-vectors-negative300.bin'
glove = 'glove.6B/glove.6B.50d.word2vec'


with open('../data/compounds_list_all.json') as input:
    all_compounds = json.load(input)['compounds']
    filename = MODEL_PATH + glove
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    list_glove = []
    for compound in all_compounds:
        if compound in model.index2word:
            list_glove.append(compound)

    filename = MODEL_PATH + word2vec
    model = KeyedVectors.load_word2vec_format(filename, binary=True)
    list_w2v = []
    for compound in all_compounds:
        if compound in model.index2word:
            list_w2v.append(compound)

    with open('../data/compounds.json', 'w') as output:
        json.dump({'glove': list_glove,
                'word2vec': list_w2v}, output)
