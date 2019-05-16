import datetime
import json
import math
import sys
from random import shuffle

import numpy as np
import matplotlib.pyplot as plt

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, LSTM, RepeatVector
from gensim.models import KeyedVectors


from data_generator import DataGenerator
from networks import get_LSTM_v1


MORPHOLOGICAL_DERIVATIONS = 'morphological_derivations'
COMPOUNDS = 'compounds'


''' 
Important:
model.wv.index2word[i] returns the i'th word 
model.wv[word] returns the corresponding embedding for the word
'''


"""-------Training Configuration-------"""

MODEL_PATH = '../data/embedding_models/'
word2vec = 'GoogleNews-vectors-negative300.bin'
glove = 'glove.6B/glove.6B.50d.word2vec'

train_test_split = 0.5
batch_size = 1000  # max size of the training set
n_epochs = 1000
cpu_cores = 12 # multiprocessing.cpu_count()
embedding_model_to_use = 'glove'

"""-------Training Configuration-------"""


STARS = '**********'
np.set_printoptions(threshold=sys.maxsize)
date = datetime.datetime.now()


def data_generator_test():
    print('Data Generator check:')
    first_batch = training_generator.__getitem__(2)
    print('Shape of X and Y:')
    print('batch size, embed dim')
    print(first_batch[0].shape)  # batch size, hot_enc dim
    print('batch size, word length, hot_enc dim')
    print(first_batch[1].shape)  # batch size, word length, hot_enc dim
    print(STARS)


def oneHotEncoder_test(generator):
    print('Hot Encoding example:')
    print('Transform:')
    s = ['hallo']
    print(s)
    print('to One Hot Encoding and back to:')
    seq = generator.word_2_seq_hot_enc(s)
    word = generator.seq_hot_enc_2_word(seq)
    print(word)
    print(STARS)


def get_partition(embedding_model, style, percent):
    validation_tokens_list = []
    if style == COMPOUNDS:
        with open('../data/compounds_separated.json') as input:
            compounds = json.load(input)[embedding_model_to_use]
            shuffle(compounds)  # shuffle to get random subset in the next step
            validation_tokens_list = compounds[math.ceil((1-percent)*len(compounds)):]

    elif style == MORPHOLOGICAL_DERIVATIONS:
        with open('../data/morphological_derivations_separated.json') as input:
            derivations = json.load(input)
            prefix_tokens = derivations[embedding_model_to_use]['prefix']
            for token in prefix_tokens:
                validation_tokens_list.append(token)

            suffix_tokens = derivations[embedding_model_to_use]['suffix']
            for token in suffix_tokens:
                validation_tokens_list.append(token)

            shuffle(validation_tokens_list)
            validation_tokens_list = validation_tokens_list[math.ceil((1 - percent) * len(validation_tokens_list)):]

    # create a list of the ids corresponding to the words to exclude
    validation_id_list = []
    for validation_token in validation_tokens_list:
        validation_id_list.append(embedding_model.vocab[validation_token].index)

    # create a training set containing all IDs and then remove all IDs of elements contained in the validation set
    train_id_list = np.arange(0, model_size)
    for id in validation_id_list:
        index_of_id = np.argwhere(train_id_list == id)
        train_id_list = np.delete(train_id_list, index_of_id)

    return {'train': train_id_list, 'validation': validation_id_list}



# load the model
if embedding_model_to_use == 'glove':
    filename = MODEL_PATH + glove
    embedding_model = KeyedVectors.load_word2vec_format(filename, binary=False)
elif embedding_model_to_use == 'word2vec':
    filename = MODEL_PATH + word2vec
    embedding_model = KeyedVectors.load_word2vec_format(filename, binary=True)


model_size = len(embedding_model.wv.index2word)
print('Embedding model: ' + embedding_model_to_use)
print('Vectors in the embedding model: ' + str(model_size))
model_name = embedding_model_to_use + '-v1_model_size=' + str(model_size) + '_epochs=' + str(n_epochs) + '_' + date.strftime("%Y-%m-%d %H:%M")
print('Corresponding files and models are stored under the name: ' + model_name)

partition = get_partition(embedding_model, MORPHOLOGICAL_DERIVATIONS, train_test_split)

for id in partition['validation']:
    if id in partition['train']:
        print('Train and validation set not disjuct, intersection: ' + embedding_model.index2word[id])

# Parameters
params = {'batch_size': batch_size,
          'shuffle': True}

# Create data generators
training_generator = DataGenerator(partition['train'], embedding_model, **params)
validation_generator = DataGenerator(partition['validation'], embedding_model, **params)

input_dim = training_generator.dim  # calculated by the DataGenerator init
hot_enc_dim = training_generator.hot_enc_len  # for the output
seq_length = training_generator.word_length  # length of the longest word of a sample from the model
# with some safety bias


data_generator_test()
oneHotEncoder_test(training_generator)


# Design the LSTM model architecture

# create LSTM
model = get_LSTM_v1(seq_length, input_dim, hot_enc_dim)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, show_shapes=True, to_file='../data/network_models/architecture_' + model_name + '.png')

# model.load_weights(filepath)

# train LSTM
history = model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    epochs=n_epochs,
    use_multiprocessing=True,
    workers=cpu_cores
)

model.save_weights('../data/network_models/' + model_name)


# make some example predictions:
Y = embedding_model.index2word[0:200]
X = embedding_model[Y]
prediction = model.predict(X)
words = training_generator.seq_hot_enc_2_word(prediction)
print(Y)
print(words)

# list all data in history
print('history data:')
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['acc'])
plt.title('loss/acc')
plt.ylabel('loss/acc')
plt.xlabel('epoch')
plt.savefig('../data/learning_curves/' + model_name + '.png')









