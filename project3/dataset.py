__doc__ = """

Common code to load sonnet and syllable data for use in multiple models.

"""
from collections import defaultdict

PUNCTUATION = "!'(),.:;?\n"

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
            words = [w.strip(PUNCTUATION) for w in ln.split(' ')]
            if not all([w in word_syllables for w in words]):
                # skip lines with invalid words. note this is probably a processing problem...
                continue
            line_trees = [[(0, words[0], 0)]]
            for w in words[1:]:
                new_line_trees = []
                for t in line_trees:
                    for syl_count in word_syllables[t[-1][1]]:
                        new_t = t + [((t[-1][0] + syl_count) % 2, w, t[-1][2] + syl_count)]
                        new_line_trees.append(new_t)
                line_trees = new_line_trees

            # find the trees that will make this line exactly 10 syllables
            for syl_count in word_syllables[w]:
                culled_line_trees = [[(stress, word) for stress, word, syl_count in tree] for tree in line_trees if tree[-1][2] == 10 - syl_count]
            if culled_line_trees:
                encoded_s.append(culled_line_trees[0])
            else:
                # didn't find any 10-line candidates. rare so just fall back to taking the first tree
                encoded_s.append([(stress, word) for stress, word, _ in line_trees[0]])
        encoded_sonnets.append(encoded_s)

    return encoded_sonnets
