# Word Embedding Space Exploration with LSTMs

## Overview

## References 
 * Main idea: Learn to Add Numbers with an Encoder-Decoder LSTM Recurrent Neural Network (https://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/)
 * different LSTM types in Keras: https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras
 * Keras data generator Example: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
 * Word2vec Keras Tutorial: https://www.tensorflow.org/tutorials/representation/word2vec
 * Word2vec Paper: Distributed Representations of Words and Phrases and their Compositionality


## Ideas and further plan
 * Which effect has the number of layers or the number of units in a LSTM network?
 * Estimate the size (layers, LSTM units) of the network based on similar NLP tasks.
 * How do design the validation/test set? Which words could be removed from the corpus without losing knowledge? 
 E.g. compound words.
 * How does the network behave (e.g. accuracy) in terms of: word length, nouns/verbs/adjectives
 