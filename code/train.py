import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, LSTM, RepeatVector

from gensim.models import KeyedVectors

from data_generator import DataGenerator
from networks import get_LSTM_v1

MODEL_PATH = '../data/embedding_models/'
word2vec = 'GoogleNews-vectors-negative300.bin'
glove = 'glove.6b/glove.6b.test50d.txt.word2vec'
train_test_split = 1
batch_size = 500  # max size of the training set
n_epochs = 5000

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

''' 
Important:
model.wv.index2word[i] returns the i'th word 
model.wv[word] returns the corresponding embedding for the word
'''


# Dataset
# load the model
filename = MODEL_PATH + glove
model = KeyedVectors.load_word2vec_format(filename, binary=False)
model_size = len(model.wv.index2word)
partition = {'train':range(0,int(train_test_split*model_size)),
             'validation':range(int(train_test_split*model_size), int(model_size))}
labels = model.wv

model_name = 'v1_model_size=' + str(model_size) + '_epochs=' + str(n_epochs) + '_' + date.strftime("%Y-%m-%d %H:%M")

# Parameters
params = {'batch_size': batch_size,
          'shuffle': True}
# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

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

# model.load_weights(filepath)

# train LSTM
history = model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    epochs=n_epochs
)

model.save_weights('../data/network_models/' + model_name)


# make some example predictions:
Y = labels.index2word[0:200]
X = labels[Y]
prediction = model.predict(X)
words = training_generator.seq_hot_enc_2_word(prediction)
print(Y)
print(words)

# list all data in history
print('history data:')
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

plt.savefig('../data/learning_curves/' + model_name + '.png')


