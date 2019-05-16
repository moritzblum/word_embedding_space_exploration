import json
import os

print(os.path.abspath(__file__))
with open('../data/morphological_derivations_glove.json') as glove:
    g = json.load(glove)
    with open('../data/morphological_derivations_word2vec.json') as word2vecve:
        w = json.load(word2vecve)
        with open('../data/morphological_derivations.json', 'w') as out:
            json.dump({'glove': g,
                       'word2vec': w}, out)