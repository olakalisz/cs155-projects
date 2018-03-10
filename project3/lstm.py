from collections import defaultdict
from keras.models import Sequential
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import LSTM, Dense, Activation
import argparse
import itertools
import numpy as np
import random
import sys

def load_sonnets():
    with open('data/shakespeare.txt') as f:
      lines = [line.strip(' ').lower() for line in f]
    sonnets = []
    ln_start = 0
    ln_end = 0
    for ln, content in enumerate(lines):
        if content[:-1].isdigit():
            ln_start = ln + 1
        elif not content[:-1]:
            if ln - 1 == ln_end:
                sonnets.append(lines[ln_start:ln_end + 1])
        elif ln + 1 == len(lines):
            sonnets.append(lines[ln_start:ln_end + 1])
        else:
            ln_end = ln

    return sonnets


def load_syllables():
    words = []
    syllable_dict = {}
    word_syllable_dict = {}
    rev_syllable_dict = defaultdict(list)
    rev_end_syllable_dict = defaultdict(list)

    with open('data/Syllable_dictionary.txt') as f:
        for i, line in enumerate(f):
            tokens = line.strip().split(' ')
            words.append(tokens[0])
            syllable_dict[i] = []
            for syl in tokens[1:]:
                if syl[0] == 'E':
                    rev_end_syllable_dict[int(syl[1:])].append(i)
                    syllable_dict[i].append(int(syl[1:]))
                else:
                    rev_syllable_dict[int(syl)].append(i)
                    syllable_dict[i].append(int(syl))
            word_syllable_dict[tokens[0]] = syllable_dict[i]

    return words, syllable_dict, word_syllable_dict, rev_syllable_dict, rev_end_syllable_dict


def stress_repr(sonnets, word_syllables):
    """Generate the stress representation for the given sonnets.

    For each line of words, turn it into a list of tuples:
        (first_syllable_stress, word)

    Additionally, produce a stress mapping for each word:
        TODO

    """
    encoded_sonnets = []
    for s in sonnets:
        encoded_s = []
        for ln in s:
            words = ln.strip('\n,?:').split(' ')
            line_trees = [[(0, words[0], 0)]]
            for w in words[1:]:
                new_line_trees = []
                for t in line_trees:
                    print(t)
                    for syl_count in word_syllables[t[-1][1]]:
                        print(syl_count)
                        new_t = t + [((t[-1][0] + syl_count) % 2, w, t[-1][2] + syl_count)]
                        print(new_t)
                        new_line_trees.append(new_t)
                        print(new_line_trees)
                line_trees = new_line_trees

            for syl_count in word_syllables[w]:
                culled_line_trees = [[(stress, word) for stress, word, syl_count in tree] for tree in line_trees if tree[-1][2] == 10 - syl_count]
                print('CULLED', culled_line_trees)
            encoded_s.append(culled_line_trees[0])
        encoded_sonnets.append(encoded_s)

    return encoded_sonnets


def _sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def construct_lstm_model(maxlen=40, initial_weights=None):
    """Construct a 2-layer LSTM model on the sonnets dataset.

    Optionally, load from saved weights by passing a filename to initial_weights.

    """
    sonnets = load_sonnets()
    chars = sorted(set([c for s in sonnets for l in s for c in l]))

    model = Sequential()
    model.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    if initial_weights:
        model.load_weights(initial_weights)

    return model


def train_lstm(model, initial_epoch=0, epochs=600):
    def _on_epoch_end(epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    sonnets = load_sonnets()
    chars = sorted(set([c for s in sonnets for l in s for c in l]))
    text = ''.join([c for s in sonnets for l in s for c in l])
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    print_callback = LambdaCallback(on_epoch_end=_on_epoch_end)
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}-2layer.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    model.fit(x, y,
              batch_size=128,
              epochs=args.epochs,
              callbacks=[checkpoint],
              initial_epoch=args.initepoch)


def generate_sonnet(model, maxlen=40, sonnets=None, chars=None):
    """Generate a sonnet with the given model.

    Uses the trained model to predict a new sonnet and outputs it to stdout.

    """
    sonnets = sonnets or load_sonnets()
    chars = chars or sorted(set([c for s in sonnets for l in s for c in l]))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    text = ''.join([c for s in sonnets for l in s for c in l])
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.25, 0.75, 1.5]:
        print('----- temperature:', diversity)

        generated = ''
        sentence = "shall i compare thee to a summer's day?\n"
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(670):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = _sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--action', type=str, choices=['train', 'generate'], required=True)
    parser.add_argument('--initepoch', type=int)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()

    model = construct_lstm_model(initial_weights=args.weights)

    if args.action == 'train':
        train_lstm(model, args.initepoch, args.epochs)
    elif args.action == 'generate':
        generate_sonnet(model)
