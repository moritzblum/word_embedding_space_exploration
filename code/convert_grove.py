# The models provided by stanford (glove) mast be converted to a different ".word2vec" format before using them with gensim.
from gensim.scripts.glove2word2vec import glove2word2vec
filename= 'glove.6b.200d.txt'
glove_input_file = '../data/embedding_models/glove.6B/' + filename
word2vec_output_file = '../data/embedding_models/glove.6B/' + filename + '.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)