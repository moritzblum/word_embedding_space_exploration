import time

import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, shuffle=True):
        """
        Works as a data generator implementing the Keras interface. Provides batches for training. The correctness of an
        generated batch can be checked by printing one sample batch: call __getitem__(4) on a correct created
        DataGenerator instance.
        (ID is an identifier for a given sample of the dataset: enumeration of the gesnim/file list)
        :param list_IDs: separation of the source dataset into train and validation. An index defines for each object
        the membership.
        :param labels: Set containing the data. X[i] = labels[word],  Y[i] = word = labels.index2word[ID]
        :param batch_size:
        :param shuffle:
        """
        self.dim = len(labels[labels.index2word[0]])
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

        # set up the encoder with all the possible chars
        self.eosTag = '#'  # because it is not contained in the corpus
        self.enc = OneHotEncoder(handle_unknown='ignore')
        i = 0
        chars = []
        for word in labels.index2word:
            i += 1
            if i == 900:  # the first 900 word should contain all possible chars
                break
            else:
                for c in word:
                    if [c] not in chars:
                        chars.append([c])
        chars.append([self.eosTag])
        self.hot_enc_len = len(chars)
        self.enc.fit(chars)
        self.word_length = 80

    def on_epoch_end(self):
        """Updates indexes after each epoch. Implements the shuffle functionality."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates batches of data with one hot encoding and padding for sequence generation
        :param list_IDs_temp:
        :return: tuple of (X,Y), where X is of shape (batch_size, word_emb_len) and
        Y is of shaoe  (batch_size, word_length, hot_enc_len)
        """
        X = np.empty((self.batch_size, self.dim))
        Y = np.empty((self.batch_size, self.word_length, self.hot_enc_len))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):  # The enumerate() function adds a counter to an iterable.
            word = self.labels.index2word[ID]
            # Store sample
            X[i, ] = self.labels[word]
            # Store class
            char_hot_enc_pad = self.word_2_seq_hot_enc_sample(word)
            Y[i] = char_hot_enc_pad
        return X.reshape(self.batch_size, self.dim), Y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def word_2_seq_hot_enc_sample(self, word):
        char_list = np.expand_dims(list(word + self.eosTag), axis=1)
        char_hot_enc = self.enc.transform(char_list).toarray()
        pad = np.zeros((self.word_length - len(char_hot_enc), self.hot_enc_len))
        char_hot_enc_pad = np.concatenate((char_hot_enc, pad))
        return char_hot_enc_pad

    def word_2_seq_hot_enc(self, word_batch):
        batch_encodings = []
        for word in word_batch:
            batch_encodings.append(self.word_2_seq_hot_enc_sample(word))
        return batch_encodings

    def seq_hot_enc_2_word(self, seq_hot_enc_batch):
        """Invert the hot encoding for a batch of sequences containing these hot enc vectors."""
        words = []
        for seq in seq_hot_enc_batch:
            word = self.enc.inverse_transform(seq)  # chooses the entry with the largest value, in the case that two
            # entries have the same value, it chooses the fist entry
            word = word.reshape(self.word_length)
            word_string = ''
            for char in word:
                if char is not None:
                    word_string = word_string + char
                else:
                    word_string = word_string + '0'
            words.append(word_string)
        return words


