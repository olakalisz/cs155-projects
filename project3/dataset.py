__doc__ = """

Common code to load sonnet and syllable data for use in multiple models.

"""
from collections import defaultdict


def load_sonnets():
    """Load normalized sonnets into a single list.

    Loads all sonnets from the given data file, normalizes them to lowercase with no trailing
    whitespace, and returns a list of sonnets, which are each a list of lines.

    """
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
    """Load syllable data and preprocess it into word and syllable dictionaries.

    Returns:
        words, a simple list of words in the order they appear in the data file.
        syllable_dict, a dictionary of syllable data (i.e. a list of possible number of syllables),
            keyed on word index for each word in the data file
        word_syllable_dict, a dictionary of the same data as syllable_dict, but keyed on word string
            rather than index.
        rev_syllable_dict, a dictionary from number of syllables to lists of all word indexes that
            possibly contain that number of syllables
        rev_end_syllable_dict, like rev_syllable_dict but only for end syllables

    """
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
