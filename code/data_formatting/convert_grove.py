from gensim.scripts.glove2word2vec import glove2word2vec
import os

""" The models provided by stanford (glove) mast be converted to a different ".word2vec" format before using them with gensim."""

def convert_all():
    for file in os.listdir('../data/embedding_models/glove.6B/'):
        if not file.startswith('.'):
            glove_file = '../data/embedding_models/glove.6B/' + file
            glove2word2vec(glove_file, glove_file[:-4] + '.word2vec')
            print(file + ' - converted')

convert_all()