import json

from gensim.models import KeyedVectors

"""
Reads in a list of onomatopoeia words written linewise in a .txt file, filters out all words which are not contained in 
the word embedding dataset, creates a json and dumps it to file. 

This script only handles the glove dataset!!!
"""

MODEL_PATH = '../../data/embedding_models/'
word2vec = 'GoogleNews-vectors-negative300.bin'
glove = 'glove.6B/glove.6B.50d.word2vec'

with open('../../data/onomatopoeia_list.txt') as input:
    filename = MODEL_PATH + glove
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    list_glove = []
    for compound in input.readlines():
        compound = compound[:-1]
        print(compound)
        if compound in model.index2word:
            list_glove.append(compound)

with open('../../data/onomatopoeia_glove.json', 'w') as output:
    json.dump({'glove': list_glove}, output)
