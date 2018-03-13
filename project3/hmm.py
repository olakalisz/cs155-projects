#!/usr/bin/python

import argparse
import itertools
from hmmlearn import hmm
from nltk.corpus import cmudict
import numpy as np

from dataset import PUNCTUATION, load_sonnets, load_syllables, stress_repr

def get_word_mapped_lines(sonnets=None, words=None, line_model=False):
    """Returns sonnets formatted as word indexes rather than character data.

    Given sonnets presented as a lists of lines of character data, parse the words out and map them
    to integer indices using the provided word lists.

    If sonnets and words are not provided, they will be loaded from the default files.

    If line_model is True, the model is trained to generate single lines instead of entire sonnets.

    Returns: 
        sonnets, a 2D numpy array of sonnet lines, where each element is either a word index or an
            end of line marker, eol_marker.
        word_indicies, a dictionary from character data for a word to its index in words.
        eol_marker, the special integer that is used to indicate the end of a line

    """
    sonnets = sonnets or load_sonnets()
    if not words:
        words, _, _, _, _ = load_syllables()
    # add a word for newlines so that the HMM can learn to do when a line break
    words.append('\n')
    word_indices = {c: i for i, c in enumerate(words)}
    eol_marker = word_indices['\n']
    X = []
    for s in sonnets:
        smapped = []
        for l in s:
            if line_model:
                smapped = []
            syl_count = 0
            ws = l.split(' ')
            for w in ws:
                if w not in words:
                    w = w.rstrip("!'(),.:;?\n")
                    if w not in words:
                        w = w.strip("!'(),.:;?\n")
                wi = word_indices[w]
                smapped.append(wi)
            if line_model:
                X.append(smapped)
            else:
                smapped.append(eol_marker)
        if not line_model:
            X.append(smapped)

    # hack attack: add some words that are never seen to the last line of input to make it a full
    # multinomial so that hmmlearn doesn't break
    all_words = set(list(range(len(words) + 1)))
    found_words = set()
    for ln in X:
        found_words = found_words.union(ln)
    missing_words = all_words.difference(found_words)
    X[-1].extend(list(missing_words))

    # make the list not ragged, with EOL markers
    max_words = max([len(s) for s in X])
    for i in range(len(X)):
        X[i] += [eol_marker] * (max_words + 1 - len(X[i]))

    X = np.array(X)

    return X, word_indices, eol_marker


def get_stress_mapped_lines(encoded_sonnets, words):
    """Returns sonnets formatted as word indexes with stress data.

    Given sonnets encoded as in dataset.stress_repr, map them to integers of:
        word_idx << 1 + stress,
    and use a hack to get a valid multinomial distribution to return in a 2D numpy array.

    Returns: 
        sonnets, a 2D numpy array of sonnet lines, where each element is either a word index or an
            end of line marker, eol_marker.
        word_indicies, a dictionary from character data for a word to its index in words.
        eol_marker, the special integer that is used to indicate the end of a line

    """
    words.append('\n')
    word_indices = {c: i for i, c in enumerate(words)}
    eol_marker = (word_indices['\n'] << 1) - 1
    X = []
    for s in encoded_sonnets:
        for l in s:
            X.append([(word_indices[word] << 1) + stress for stress, word in l] + [eol_marker])

    # hack attack: add some words that are never seen to the last line of input to make it a full
    # multinomial so that hmmlearn doesn't break
    all_words = set(list(range(len(words) * 2 - 1)))
    found_words = set()
    for ln in X:
        found_words = found_words.union(ln)
    missing_words = list(all_words.difference(found_words))
    X[-1].extend(missing_words)

    # make the list not ragged, with EOL markers
    max_words = max([len(s) for s in X])
    for i in range(len(X)):
        X[i] += [eol_marker] * (max_words + 1 - len(X[i]))

    X = np.array(X)
    print(X.shape)

    return X, word_indices, eol_marker


def train_hmm(X, n_components=10, n_iter=100):
    model = hmm.MultinomialHMM(n_components=n_components, n_iter=n_iter)
    model.fit(X)

    return model


def generate_sonnet(model, eol_marker, line_model=False):
    """Generate a sonnet with the given model.

    If line_model is false, does a single sampling from the HMM which is assumed to be trained on
    entire sonnets. Otherwise, takes 14 independent line samples from the model trained on lines in
    isolation.

    eol_marker is the special index used to indicate the end of a line.

    Returns a list of lines of word indicies.

    """
    sonnet = []
    if line_model:
        for i in range(14):
            X_gen, states = model.sample(15)
            X_gen = list(X_gen.flatten())
            if eol_marker in X_gen:
                # in case there was a low probability transition away from EOL back to word data
                X_gen = X_gen[:X_gen.index(eol_marker) - 1]
            sonnet.append(X_gen)
    else:
        X_gen, states = model.sample(210)
        X_gen = list(X_gen.flatten())
        while eol_marker in X_gen:
            eol_idx = X_gen.index(eol_marker)
            if eol_idx > 0:
                sonnet.append(X_gen[:eol_idx])
            X_gen = X_gen[X_gen.index(eol_marker) + 1:]
        sonnet.append(X_gen)

    return sonnet


MODEL_TYPES = [
    'line', # generate a line at a time, independently
    'sonnet', # generate whole sonnets
    'stress', # generate a line at a time, with stress encoded
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=MODEL_TYPES, required=True)
    args = parser.parse_args()

    sonnets = load_sonnets()
    words, _, word_syllables, _, _ = load_syllables()
    if args.model == 'stress':
        encoded_sonnets = stress_repr(sonnets, word_syllables)
        X, word_indices, eol_marker = get_stress_mapped_lines(encoded_sonnets, words=words)
        word_lookup = lambda i: words[i >> 1]
    else:
        X, word_indices, eol_marker = get_word_mapped_lines(words=words, line_model=args.model=='line')
        word_lookup = lambda i: words[i]
    model = train_hmm(X)
    sonnet = generate_sonnet(model, eol_marker, line_model=args.model!='line')

    for ln in sonnet:
        print(' '.join([word_lookup(i) for i in ln]).capitalize())
