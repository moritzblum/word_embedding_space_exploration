# Word Embedding Space Exploration with LSTMs


## Project Description

### Motivation and Idea

Word embeddings are a set of language modeling techniques in natural language processing (NLP) where words or phrases 
from the vocabulary are mapped to vectors. One method to generate such a mapping are through neural networks which are 
used in [Word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).
The objective is that words with a similar context lie close together in space. Another important fact is, that they
are reducing the number of dimensions in comparison to one hot encoding of words. Because of these great properties, they are used in many different NLP tasks and 
have shown to be a good working word representation. But these mappings are from a discrete word
space to another discrete vector space and new fantasy words with an understandable and derivable meaning have no associated vector 
in the word embedding space, because these were never learned. The other way around, from word embeddings to fantasy words the same.

The goal of this project is to verbalize points in the word embedding space, which were not trained. Doing this will give us deeper 
insights into the space of word embeddings e.g. in terms of word locations and could have multiple applications like
e.g. the understanding NN training results. 

Assuming a function exists that maps continuous vectors into words, it should be 
possible to learn this function with a neural network, since these can approximate any possible function arbitrary close. 
If you then enter a new vector into the function, the output should be as meaningful as possible. 
For example, a vector between book and poster could output bookster or any other meaningful mixture of these words. 
Furthermore, the distance between the two words could be continuously traversed and one should then be able to observe a 
fluid change in the output word that is related to the position.
In addition, word vectors can be added and then verbalized whose combination actually had no verbalized meaning.


### Setup
 
To verbalize word embedding vectors, a function that maps high dimensional vectors to strings is required. Since the 
words are different in length, a sequence-generation RNN (one-to-many) with LSTM cells can be used. 
In this project the decision was made in LSTM cells, but with the similar GRUs one would expect the same behavior. 

Input to the network are the raw word embedding vectors
and the output is a sequence of vectors, where each vector is a char in one-hot encoding. Each word ends with a EOS (end of sequence)
token. For training the Stanford Glove and Google pre trained Word2Vec databases are used, which contain the training data in (vector, string) tuples. 
During training, a DataGenerator generates a batch in the correct format out of these corpora. Choosing a 
validation set is not easy, so this will be discussed in more detail in the next section.
 
 
## Evaluation - How to construct validation- & test-set?
 
Simply looking at random vectors and interpreting their meaning is not simple and especially not possible to compute, 
so different approaches are required.
The goal is to find a set of words with certain properties, which ensure hopefully that
each of these words can be generated, because no important information is missing in the model.
These words should not have a unique meaning, i.e. there
are existing other words or combinations of other words with the same meaning, and should be constructable out of 
other words, e.g. "comeback" can be constructed out of "come" and "back". 
 
### Random
 
Simply selecting random words of the dataset as validation-set does not fulfill these properties. The assumption is that the
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
were extracted. This yields a [list of 848 compounds](/data/compounds_list.txt). Most of these words are contained in
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

Excluding some words which follow morphological derivations, would yield a validation set that can be reconstructed, because
the RNN can learn the semantic of the derivation rules.


### Loss and Accuracy

The loss function of the neural network is directly calculated on the vectors. One could 
think of using advanced word distances for this, but they usually do not fulfill the requirements of a loss function for NNs. 
For all trainings, the categorical crossentropy loss is used, because this is the standard classification tasks.

The accuracy is measure by comparing the input word with the output sequence until the EOS token. There are other possible 
word distances that can be used, too. 

If a model with good performance has been obtained, one can consider how to evaluate the verbalization of new word embeddings. 
But it is not sure whether it is possible to learn such a model, so this question has been postponed.



## Training

For training multiple different networks were set up and trained on the CITEC GPU-Cluster.

### Training 1

Training Configuration:
* validation = MORPHOLOGICAL_DERIVATIONS 
* train_test_split = 0.5 => Size of the validation set: 8813
* batch_size = 4000  
* n_epochs = 5000
* cpu_cores = 12 
* embedding_model_to_use = 'glove'

Network architecture: 

|Layer (type)                 |Output Shape              |Param    
|-----------------------------|--------------------------|--------|
|repeat_vector_1 (RepeatVecto |(None, 80, 50)            | 0      | 
|lstm_1 (LSTM)                |(None, 80, 15)            | 3960   | 
|lstm_2 (LSTM)                |(None, 80, 10)            | 1040   | 
|dense_1 (Dense)              |(None, 80, 70)            | 770    | 
|dense_2 (Dense)              |(None, 80, 56)            | 3976   | 

![Network Architecture](https://github.com/moritzblum/word_embedding_space_exploration/blob/master/img/architecture1.png)

* Total params: 9,746
* Trainable params: 9,746
* Non-trainable params: 0

Training results: loss: 0.2848 - acc: 0.0214 - val_loss: 0.3068 - val_acc: 0.0203

GPU cluster didn't finish training, due to time limit of 48h training. But the results are enough to make first 
reasoning. After 3000 epochs there were no improvement on training and validation loss. This is an indicator for a underfitting 
RNN model. 


### Training 2 

This time the number of LSTM layers and the number of LSTMs per layer are increased to counter underfitting by increasing the trainable model parameters, 
the training limit is extended to 72 hours and the number of epochs is reduced to ensure a correct termination of the training process.

Training Configuration:
* validation = MORPHOLOGICAL_DERIVATIONS 
* train_test_split = 0.5 => Size of the validation set: 8813
* batch_size = 4000  
* n_epochs = 4000
* cpu_cores = 4 
* embedding_model_to_use = 'glove'

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

Training results: loss: 0.2696 - acc: 0.0245 - val_loss: 0.3032 - val_acc: 0.0214

![results](https://github.com/moritzblum/word_embedding_space_exploration/blob/master/img/training_results2.png)

Again, the loss did not decrease further, which is an indicator for too much data training data and/or a too high complexity
of the learning problem.


### Training 4
To work against this underfitting, the number of model parameters was increased a second time. Now the network has as many parameters as data points in 
the training set. 

|Layer (type)                 |Output Shape              |Param    
|-----------------------------|--------------------------|---------|
|repeat_vector_1 (RepeatVecto |(None, 80, 50)            | 0       | 
|lstm_1 (LSTM)                |(None, 80, 180)           | 166320  | 
|lstm_2 (LSTM)                |(None, 80, 120)           | 144480  | 
|lstm_3 (LSTM)                |(None, 80, 120)           | 115680  | 
|dense_1 (Dense)              |(None, 80, 200)           | 24200   | 
|dense_2 (Dense)              |(None, 80, 56)            | 11256   | 

Total params: 461,936
Trainable params: 461,936
Non-trainable params: 0


Training results: loss: 0.2412 - acc: 0.0315 - val_loss: 0.3420 - val_acc: 0.0213

But again, the loss stopped to decrease after some time.

The assumption is that a much larger model is required, which requires significantly more computing power. But it is not 
possible to train such models with much more parameters at the moment.


## Learning Process Validation

In order to avoid that the training was error-prone and our results are not correct, the training was performed on a smaller amount of data. 
If it is possible to learn a small set of words by heart, the network architecture and the learning algorithm are 
working correct.

### Onomatopoeia

The idea is to use onomatopoeia
words, because they are very simple built up and should do to this be easier to learn. Omatopoeia are words that 
phonetically imitates, resembles, or suggests the sound that they describe. Some example words are extracted from 
this [website](https://www.noisehelp.com/examples-of-onomatopoeia.html).


|Layer (type)                 |Output Shape              |Param    
|-----------------------------|--------------------------|---------|
|repeat_vector_1 (RepeatVecto |(None, 80, 50)            | 0       | 
|lstm_1 (LSTM)                |(None, 80, 80)           | 41920  | 
|lstm_2 (LSTM)                |(None, 80, 60)           | 33840  | 
|lstm_3 (LSTM)                |(None, 80, 80)           | 45120  | 
|dense_1 (Dense)              |(None, 80, 100)           | 8100   | 
|dense_2 (Dense)              |(None, 80, 56)            | 5656   | 

Total params: 134,636
Trainable params: 134,636
Non-trainable params: 0

Training results: 1s 504ms/step - loss: 5.2252e-06 - acc: 0.0726 - val_loss: 0.6006 - val_acc: 0.0310

If you now look at the training results, you can see that the loss is very small, but the acc has a much higher value. This indicates, that
the training was successful, but the accuracy measure is more sensitive against errors in the output. It's not exactly clear why this is the case.
Maybe the error adds up across in the word length. All previous trainings were conducted with a word length of 80 characters, 
as there were also very long words in the training data set, but for this training, all words have less than 10 characters.
=> Complexity way to high and smaller word length is sufficient in this case.

Training with word length 10 and without a validation set:

Training results for words with word length of 10:0s 267ms/step - loss: 4.8331e-04 - acc: 0.5820

To verify that the results are really good, you can look at the prediction for random data from the training data set.

label, prediction:
* ('cluck', 'cluck#####'), 
* ('hum', 'hum#######'), 
* ('yap', 'yap#######'), 
* ('lisp', 'lisp######'), 
* ('snip', 'snip######'), 
* ('sputter', 'sputter###'), 
* ('rattle', 'rattle####'), 
* ('whomp', 'whomp#####'), 
* ('bawl', 'bawl######'), 
* ('trickle', 'trickle###'), 
* ('rap', 'rap#######'), 
* ('whoosh', 'whoosh####'), 
* ('belch', 'belch#####')

This are in fact very good results. The network seems to be able to recunstruct each word out of its word embedding.
Again it is strange that the accuracy is not at 1, because the results seem quite perfect.

Now one can check, how the training behave in case of a random validation set, as indicated in the section Evaluation.
Therefore a training was set up with a split in training and validation set of 70% against 30%:

Training results: 0s 284ms/step - loss: 3.6720e-04 - acc: 0.5720 - val_loss: 5.0976 - val_acc: 0.2300

Prediction of words of the validation set:

* ('munch', 'cliff#####'), 
* ('snicker', 'clbbe#####'), 
* ('pop', 'pop#######'), 
* ('rap', 'woos######'), 
* ('jingle', 'muup######'), 
* ('buzz', 'buzz######'), 
* ('crash', 'twwwwhh###'), 
* ('boom', 'brup######'), 
* ('splatter', 'siiee#####'), 
* ('squish', 'plac######'), 
* ('knock', 'siffz#####'), 
* ('whistle', 'whistle###'), 
* ('honk', 'beep######'), 
* ('hum', 'hum#######'), 
* ('sputter', 'zuunzz####')

This results in a worser training accuracy than without validation. But the model must be biased and trained to the validation set, 
otherwise there is no explanation why the network is able to create this words, because the data set must be way to small 
to detect such a great behaviour. That the verbalization as indicated in the motivation does not work is obvious at this worse performance.

## Results

It could not be shown if it is possible to verbalize unknown word embeddings in a meaningful way. The problem was that the training loss 
stopped to decrease and an increase in the number of model parameters has helped little to reduce this value even further. 
With many more model parameters you might get good results, but these models are currently not trainable because of hardware 
and there would probably be new problems regarding the model size. Unfortunately, it was not possible to investigate the 
effect of the different validation sets, as successful training would have been necessary.

In order to exclude a mistake on my part, a network with a small amount of data was overfitted. 
This shows that the training was technically correct.


## Important links 
 * [Keras data generator Example](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
 * [Word2vec Keras Tutorial](https://www.tensorflow.org/tutorials/representation/word2vec)


