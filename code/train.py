import datetime
import json
import math
import sys
from random import shuffle

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
from gensim.models import KeyedVectors

from data_generator import DataGenerator
from networks import *

MORPHOLOGICAL_DERIVATIONS = 'morphological_derivations'
COMPOUNDS = 'compounds'
ONOMATOPOEIA = 'onomatopoeia'


''' 
Training of a LTSM network.

Important:
model.wv.index2word[i] returns the i'th word 
model.wv[word] returns the corresponding embedding for the word
'''


"""-------Training Configuration-------"""

MODEL_PATH = '../data/embedding_models/'
word2vec = 'GoogleNews-vectors-negative300.bin'
glove = 'glove.6B/glove.6B.50d.word2vec'

# specify which LSTM model to use in line 208

validation = ONOMATOPOEIA # MORPHOLOGICAL_DERIVATIONS  # COMPOUNDS
train_test_split = 0.3
batch_size = 100  # max size of the training set
n_epochs = 2000
cpu_cores = 2
embedding_model_to_use = 'glove'
multipe_gpu = False


"""-------Training Configuration-------"""


STARS = '**********'
np.set_printoptions(threshold=sys.maxsize)
date = datetime.datetime.now()
size_validation_set = 1

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
    if style == ONOMATOPOEIA:
        """ Splits up the ONOMATOPOEIA in training and validation set. """
        with open('../data/onomatopoeia_glove.json') as input:
            onomatopoeia = json.load(input)[embedding_model_to_use]
            shuffle(onomatopoeia)  # shuffle to get random subset in the next step
            validation_tokens_list = onomatopoeia[math.ceil((1-percent)*len(onomatopoeia)):]
            train_tokens_list = onomatopoeia[:math.ceil((1-percent)*len(onomatopoeia))]
            while True:
                if len(validation_tokens_list) < batch_size:
                    print('validation_tokens_list smaller than one batch -> extend size by adding some of them '
                          'multiple times. Size validation: '
                          + str(len(validation_tokens_list))
                          + 'Size of batch: ' + str(batch_size))
                    validation_tokens_list = np.concatenate((validation_tokens_list,
                                                             validation_tokens_list[:batch_size - len(validation_tokens_list)]))
                else:
                    break
            while True:
                if len(train_tokens_list) < batch_size:
                    print('validation_tokens_list smaller than one batch -> extend size by adding some of them '
                          'multiple times. Size validation: '
                          + str(len(train_tokens_list))
                          + 'Size of batch: ' + str(batch_size))
                    train_tokens_list = np.concatenate((train_tokens_list,
                                                             train_tokens_list[:batch_size - len(train_tokens_list)]))
                else:
                    break

        size_validation_set = len(validation_tokens_list)
        size_training_set = len(train_tokens_list)
        print('Size of the validation set: ' + str(size_validation_set))
        print('Size of the training set: ' + str(size_training_set))
        # create a list of the ids corresponding to the words to exclude
        validation_id_list = []
        for validation_token in validation_tokens_list:
            validation_id_list.append(embedding_model.vocab[validation_token].index)

        training_id_list = []
        for training_token in train_tokens_list:
            training_id_list.append(embedding_model.vocab[training_token].index)

        return {'train': training_id_list, 'validation': validation_id_list}
    else:

        if style == COMPOUNDS:
            with open('../data/compounds.json') as input:
                compounds = json.load(input)[embedding_model_to_use]
                shuffle(compounds)  # shuffle to get random subset in the next step
                validation_tokens_list = compounds[math.ceil((1-percent)*len(compounds)):]
                while True:
                    if len(validation_tokens_list) < batch_size:
                        print('validation_tokens_list smaller than one batch -> extend size by adding some of them '
                              'multiple times. Size validation: '
                              + str(len(validation_tokens_list))
                              + 'Size of batch: ' + str(batch_size))
                        validation_tokens_list = np.concatenate((validation_tokens_list,
                                                                 validation_tokens_list[:batch_size - len(validation_tokens_list)]))
                    else:
                        break
                print('New validation set size: ' + str(len(validation_tokens_list)))
        elif style == MORPHOLOGICAL_DERIVATIONS:
            with open('../data/morphological_derivations_glove.json') as input:
                derivations = json.load(input)
                prefix_tokens = derivations[embedding_model_to_use]['prefix']
                for token in prefix_tokens:
                    validation_tokens_list.append(token)

                suffix_tokens = derivations[embedding_model_to_use]['suffix']
                for token in suffix_tokens:
                    validation_tokens_list.append(token)

                shuffle(validation_tokens_list)
                validation_tokens_list = validation_tokens_list[math.ceil((1 - percent) * len(validation_tokens_list)):]

        size_validation_set = len(validation_tokens_list)
        print('Size of the validation set: ' + str(size_validation_set))
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
model_name = embedding_model_to_use + '-v2_model_size=' + str(model_size) + '_epochs=' + str(n_epochs) + '_' + date.strftime("%Y-%m-%d %H:%M")
print('Corresponding files and models are stored under the name: ' + model_name)

partition = get_partition(embedding_model, validation, train_test_split)

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
# length of the longest word of a sample from the model with some safety bias
seq_length = training_generator.word_length  # default 80, can be adapted

data_generator_test()
oneHotEncoder_test(training_generator)


# Design the LSTM model architecture

# create LSTM
model = get_LSTM_v4(seq_length, input_dim, hot_enc_dim)
try:
    if multipe_gpu:
        model = multi_gpu_model(model, gpus=2)
except ValueError:
    print('No multi-gpu usage, because of an multi_gpu_model Error.')
    

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# plot_model(model, show_shapes=True, to_file='../data/network_models/architecture_' + model_name + '.png')

# model.load_weights('../data/network_models/' + 'glove-v1_model_size=400000_epochs=3000_2019-06-15 11:27')

# train LSTM
history = model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    epochs=n_epochs,
    validation_steps=size_validation_set/batch_size,
    use_multiprocessing=True,
    workers=cpu_cores
)

model.save_weights('../data/network_models/' + model_name + '.h5')


# make some example predictions:
Y = []
for index in partition['validation']:
    Y.append(embedding_model.index2word[index])
X = embedding_model[Y]
prediction = model.predict(X)
words = training_generator.seq_hot_enc_2_word(prediction)

test_tuples = []
index = 0
for test_element in Y:
    test_tuples.append((Y[index], words[index]))
    index += 1
print(test_tuples)

# list all data in history
print('history data:')
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['acc'], label='acc')
#plt.plot(history.history['val_loss'], label='val_loss')
#plt.plot(history.history['val_acc'], label='val_acc')
plt.legend()
plt.title('loss/acc')
plt.ylabel('loss/acc')
plt.xlabel('epoch')
plt.savefig('../data/learning_curves/' + model_name + '_2_' + '.png')









