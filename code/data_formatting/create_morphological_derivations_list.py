import json

from gensim.models import KeyedVectors

MODEL_PATH = '../../data/embedding_models/'
word2vec = 'GoogleNews-vectors-negative300.bin'
glove = 'glove.6B/glove.6B.50d.word2vec'


def find_words(model):
    example_suffixes = ['ness', 'ise', 'ish', 'ly', 'al', 'fy', 'able', 'ance', 'er']
    example_prefix = ['re', 'un', 'en', 'be']

    words_contained_with_and_without_suffix = []
    words_contained_with_and_without_prefix = []

    for word in model.index2word:
        for suffix in example_suffixes:
            if word.endswith(suffix):
                word_without_suffix = word[:-len(suffix)]
                if word_without_suffix in model.index2word:
                    words_contained_with_and_without_suffix.append(word)
                    break

        for prefix in example_prefix:
            if word.startswith(prefix):
                word_without_prefix = word[len(prefix):]
                if word_without_prefix in model.index2word:
                    words_contained_with_and_without_prefix.append(word)
                    break


    print(filename)

    print('Suffix list:')
    print(len(words_contained_with_and_without_suffix))

    print('Prefix list:')
    print(len(words_contained_with_and_without_prefix))

    return {'suffix': words_contained_with_and_without_suffix, 'prefix': words_contained_with_and_without_prefix}


# glove
filename = MODEL_PATH + glove
model = KeyedVectors.load_word2vec_format(filename, binary=False)
word_dict = find_words(model)
with open('../../data/morphological_derivations_glove.json', 'w') as out_file:
    json.dump(word_dict, out_file)

# word2vec
filename = MODEL_PATH + word2vec
model = KeyedVectors.load_word2vec_format(filename, binary=True)
word_dict = find_words(model)
with open('../../data/morphological_derivations_word2vec.json', 'w') as out_file:
    json.dump(word_dict, out_file)
