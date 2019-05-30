# Word Embedding Space Exploration with LSTMs

## Overview

## Project Description

## References 
 * Main idea: [Learn to Add Numbers with an Encoder-Decoder LSTM Recurrent Neural Network](https://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/)
 * [different LSTM types in Keras](https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras)
 * [Keras data generator Example](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
 * [Word2vec Keras Tutorial](https://www.tensorflow.org/tutorials/representation/word2vec)
 * Word2vec Paper: Distributed Representations of Words and Phrases and their Compositionality
 * Google translator seq2seq paper: Google’s Neural Machine Translation System: Bridging the Gap between Human and 
 Machine Translation
 


## Ideas and further plan
 * How does the network behave (e.g. accuracy) in terms of: word length, nouns/verbs/adjectives
 
 
 ## Infos
 * Google translator uses ~8 layers per encoding and decoding
 * Pytorch TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION example used one layer each with 256 units
 * "2 layers are nice, 5 layers are better and 7 layers are very hard to train"
 
 
 ## Motivation
 
 Verbalization of points in the word embedding space, which were not trained. 
 
 
 ## Setup
 
 
 ## Evaluation - How to construct validation- & test-set?
 
 The goal is to find a set of words with certain properties, which ensure hopefully that
 each of these words can be generated, because no important information is missing in the model.
 These words should not have a unique meaning, i.e. there
 are existing other words or combinations of other words with the same meaning , and should be constructable out of 
 other words, e.g. "comeback" can be constructed out of "come" and "back". 
 
 ### Random
 
 Simply selecting random words of the dataset as test-set does not fulfill these properties. The assumption is that the
 RNN will not be able to reconstruct these words correctly out of the embeddings, because without the fulfilled 
 properties, no information about most of these words is contained in the trained model.
 
 ### Compounds
 
 The idea was to design a test set based on compounds (e.g., Computerlinguistik 'computational linguistics'). Assuming 
 that a not minor subset exists, where not only the string is constructed out of other strings, but as well the meaning 
 is represented by the other words, no information is missing in the corpus and the trained system should be able to 
 reconstruct these words. 

 But compounds are a special case in German, because  German writes compound nouns without spaces, different than e.g. 
 in English.  Due to this fact, there is not much research on this topic for English and no large corpus exists. To create a test 
 set, all words of a [website with collected compounds](https://www.learningdifferences.com/Main%20Page/Topics/Compound%20Word%20Lists/Compound_Word_%20Lists_complete.htm) 
 were extracted. This yields a [list of 848 compounds](/data/compounds_list_all.txt). Most of these words are contained in
 the trained word embeddings too. 

### Morphological Derivations

 [Morphological derivation Wikipedia](https://en.wikipedia.org/wiki/Morphological_derivation):
 
 "Derivational morphology often involves the addition of a derivational suffix or other affix."
 
 Here are examples of English derivational patterns and their suffixes:

 * adjective-to-noun: -ness (slow → slowness)
 * adjective-to-verb: -ise (modern → modernise) in British English or -ize (final → finalize) in American English and 
 Oxford spelling
 * adjective-to-adjective: -ish (red → reddish)
 * adjective-to-adverb: -ly (personal → personally)
 * noun-to-adjective: -al (recreation → recreational)
 * noun-to-verb: -fy (glory → glorify)
 * verb-to-adjective: -able (drink → drinkable)
 * verb-to-noun (abstract): -ance (deliver → deliverance)
 * verb-to-noun (agent): -er (write → writer)
 
 However, derivational affixes do not necessarily alter the lexical category; they may change merely the meaning of the
 base and leave the category unchanged. 
 
 * A prefix (write → re-write; lord → over-lord) rarely changes the lexical category in English.
 * The prefix un- applies to adjectives (healthy → unhealthy) and some verbs (do → undo) but rarely to nouns.
 * A few exceptions are the derivational prefixes en- and be-. En- (replaced by em- before labials) is usually a 
 transitive marker on verbs, but it can also be applied to adjectives and nouns to form transitive verbs: 
 circle (verb) → encircle (verb) but rich (adj) → enrich (verb), large (adj) → enlarge (verb), 
 rapture (noun) → enrapture (verb), slave (noun) → enslave (verb).

 Excluding some words which follow morphological derivations, would yield a test set that can be reconstructed, because
 the RNN can learn the semantic of the derivation rules.

### Multi-words

 Multi-words are very similar to Compounds, but the semantically couples words are separated by a whitespace. 
 Using Milti-words as test-set would require first to train word embeddings, with Multi-words as one word/vector. 
 After that, the idea is the same to the idea with Compounds.
 
 To train the word embeddings, a Multi-word tokenized text is required. NLTK provides a module 
 [nltk.tokenize.mwe module](http://www.nltk.org/api/nltk.tokenize.html?highlight=regexp%20tokenize#nltk.tokenize.regexp.RegexpTokenizer) 
 for that. The training can be done with [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) to obtain a 
 Word2Vec model. (Glove embeddings can not be trained with Gensim.)
 
 https://datascience.stackexchange.com/questions/17294/nlp-what-are-some-popular-packages-for-multi-word-tokenization



https://stackoverflow.com/questions/5532363/python-tokenizing-with-phrases

### Redundant words in English

Vocabularies that are not necessary in the english language, because e.g. there is another more frequently used word 
with the same meaing 

## Training

Training is done on the CITEC GPU-Cluster.

### Training 1

"""-------Training Configuration-------"""
validation = MORPHOLOGICAL_DERIVATIONS 
train_test_split = 0.5
batch_size = 4000  
n_epochs = 5000
cpu_cores = 12 
embedding_model_to_use = 'glove'
"""-------Training Configuration-------"""

|Layer (type)                 |Output Shape              |Param    
|-----------------------------|--------------------------|--------|
|repeat_vector_1 (RepeatVecto |(None, 80, 50)            | 0      | 
|lstm_1 (LSTM)                |(None, 80, 15)            | 3960   | 
|lstm_2 (LSTM)                |(None, 80, 10)            | 1040   | 
|dense_1 (Dense)              |(None, 80, 70)            | 770    | 
|dense_2 (Dense)              |(None, 80, 56)            | 3976   | 
Total params: 9,746
Trainable params: 9,746
Non-trainable params: 0

Size of the validation set: 8813

Training results:
* after 1000 epochs: loss: 0.2852 - acc: 0.0213 - val_loss: 0.3068 - val_acc: 0.0205
* after 2000 epochs: loss: 0.2849 - acc: 0.0213 - val_loss: 0.3073 - val_acc: 0.0205
* after 3000 epochs: loss: 0.2847 - acc: 0.0214 - val_loss: 0.3063 - val_acc: 0.0204
* after 4000 epochs: loss: 0.2846 - acc: 0.0214 - val_loss: 0.3055 - val_acc: 0.0204
*after 4502 epochs: loss: 0.2848 - acc: 0.0214 - val_loss: 0.3068 - val_acc: 0.0203

GPU cluster didn't finish training, due to time limit of 48h training. But the results are enough to make first 
reasoning. After 3000 epochs no improvement on training and validation loss. This is an indicator for a underfitting 
RNN model. 


### Training 2 

This time the number of LSTM layers and the number of LSTMs per layer are increased, the training limit is extended to
72 hours, and the number of epochs is reduced.

"""-------Training Configuration-------"""
validation = MORPHOLOGICAL_DERIVATIONS 
train_test_split = 0.5
batch_size = 4000  
n_epochs = 4000
cpu_cores = 8 
embedding_model_to_use = 'glove'
"""-------Training Configuration-------"""

|Layer (type)                 |Output Shape              |Param    
|-----------------------------|--------------------------|--------|
|repeat_vector_1 (RepeatVecto |(None, 80, 50)            | 0      | 
|lstm_1 (LSTM)                |(None, 80, 60)            | 26640  | 
|lstm_2 (LSTM)                |(None, 80, 40)            | 16160  | 
|lstm_3 (LSTM)                |(None, 80, 40)            | 12960  | 
|dense_1 (Dense)              |(None, 80, 70)            | 2870   | 
|dense_2 (Dense)              |(None, 80, 56)            | 3976   | 
Total params: 62,606
Trainable params: 62,606
Non-trainable params: 0

Size of the validation set: 8813


TODO stdout_2948_0 training history

### Training 3

Same configuration and network as in training 2 but with COMPOUNDS for validation. Due to the fact that the compounds 
set has only a size of 400 and the batch size is 4000, the validation set is extended artificially by repeating the
set multiple times.


