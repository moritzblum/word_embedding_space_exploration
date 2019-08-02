import json

from gensim.models import KeyedVectors

"""
Reads in a list of compound words written linewise in a .txt file, filters out all words which are not contained in the 
word embedding dataset, creates a json and dumps it to file. 

This script only handles glove and word2vec.
"""

MODEL_PATH = '../../data/embedding_models/'
word2vec = 'GoogleNews-vectors-negative300.bin'
glove = 'glove.6B/glove.6B.50d.word2vec'

with open('../../data/compounds_list.txt') as input:
    filename = MODEL_PATH + glove
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    list_glove = []
    for compound in input.readlines():
        if compound in model.index2word:
            list_glove.append(compound)

    filename = MODEL_PATH + word2vec
    model = KeyedVectors.load_word2vec_format(filename, binary=True)
    list_w2v = []
    for compound in input.readlines():
        if compound in model.index2word:
            list_w2v.append(compound)

    with open('../data/compounds.json', 'w') as output:
        json.dump({'glove': list_glove,
                   'word2vec': list_w2v}, output)
