import random
import re
from copy import copy
import rstr
import numpy as np
from config import Config
from PCFG import PCFG

config = Config()

pos_category_map = \
    {
        'NN': 'N', 'NNS': 'N', 'NNP': 'N', 'NNPS': 'N', 'PRP$': 'N', 'PRP': 'N', 'WP': 'N', 'WP$': 'N',
        'MD': 'V', 'VB': 'V', 'VBC': 'V', 'VBD': 'V', 'VBF': 'V', 'VBG': 'V', 'VBN': 'V', 'VBP': 'V',
        'VBZ': 'V',
        'JJ': 'J', 'JJR': 'J', 'JJS': 'J', 'LS': 'J', 'RB': 'A', 'RBR': 'A', 'RBS': 'A', 'WRB': 'A',
        'DT': 'D', 'PDT': 'D', 'WDT': 'D',
        'SYM': 'S', 'POS': 'S', '-LRB-': 'S', '-RRB-': 'S', ',': 'S', '-': 'S', ':': 'S', ';': 'S', '.': 'S',
        '``': 'S',
        '"': 'S', '$': 'S', "''": 'S', '#': 'S',
        'CD': 'C', 'DAT': 'X', 'CC': 'B', 'EX': 'E', 'FW': 'F', 'IN': 'I', 'RP': 'R', 'TO': 'T',
        'UH': 'U'
    }
pos_category_to_num = {cat: i for i, cat in enumerate(sorted(set(pos_category_map.values())))}


def generate_sentences(DATA_AMOUNT, alphabet_map):
    """Generate data and return it splitted to train, test and labels"""
    type_regex = config.Grammar.type_regex.boolean
    if type_regex:
        raw_x, raw_y = get_regex_sentences(DATA_AMOUNT, alphabet_map)
    else:
        raw_x, raw_y = get_penn_pos_data(DATA_AMOUNT, alphabet_map)

    percent_train = config.Data.percent_train.float
    num_train = int(DATA_AMOUNT * percent_train)
    zipped = list(zip(raw_x, raw_y))
    random.shuffle(zipped)
    raw_x, raw_y = zip(*zipped)
    for i, j in zip(raw_x[:5], raw_y[:5]):
        print(i, j)

    X_train = np.array(raw_x[:num_train])
    y_train = np.array(raw_y[:num_train])
    X_test = np.array(raw_x[num_train:])
    y_test = np.array(raw_y[num_train:])

    X_train, y_train = zip(*(sorted(zip(X_train, y_train), key=lambda x: len(x[0]))))
    X_test, y_test = zip(*(sorted(zip(X_test, y_test), key=lambda x: len(x[0]))))
    return X_train, y_train, X_test, y_test


def generate_random_strings(max_length, num_of_sents, alphabet):
    random_lengths = np.random.randint(low=1, high=max_length, size=num_of_sents)
    return [rstr.rstr(alphabet, length) for length in random_lengths]


def filter_out_by_allowed_pos(sent, alphabet_map):
    allowed_pos = [alphabet_map[pos] for pos in ['N', 'V', 'J', 'A', 'D', 'S', 'I', 'T']]
    for pos in sent:
        if pos not in allowed_pos:
            return False
    return True


def get_penn_pos_data(total_num_of_sents, alphabet_map):
    grammatical_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    grammaticals = []
    left = total_num_of_sents // 2
    curr_idx = 0
    while left > 0 and curr_idx < len(grammatical_sents):
        curr_grammaticals = list(grammatical_sents)[curr_idx:curr_idx + left]
        # grammaticals += list(filter(lambda x: filter_out_by_allowed_pos(x, alphabet_map), curr_grammaticals))
        grammaticals += curr_grammaticals
        curr_idx += left
        left = (total_num_of_sents // 2) - len(grammaticals)

    ungrammaticals = []
    left = total_num_of_sents // 2
    # alphabet = list(set(alphabet_map.keys()).intersection({'N', 'V', 'J', 'A', 'D', 'S', 'I', 'T'}))
    alphabet = list(set(alphabet_map.keys()))
    while left > 0:
        curr_ungrammaticals = generate_random_strings(config.Data.max_len.int, left, alphabet)
        curr_ungrammaticals = list(map(lambda sent: [alphabet_map[s] for s in list(sent)], curr_ungrammaticals))
        ungrammaticals += list(filter(lambda sent: filter_out_grammatical_sentences(sent, grammatical_sents),
                                      curr_ungrammaticals))
        left = (total_num_of_sents // 2) - len(ungrammaticals)

    data = grammaticals + ungrammaticals
    labels = np.array([1] * len(grammaticals) + [0] * len(ungrammaticals))
    return data, labels


def generate_simple_english_grammar(num_of_sents, max_seq_len):
    pcfg = PCFG.from_file('grammar.txt')
    output = set()
    while len(output) < num_of_sents:
        sent = pcfg.random_sent()
        if len(sent.split()) < max_seq_len:
            output.add(sent)
    return output


def filter_by_regex(sent, regex):
    pattern = re.compile(regex)
    if pattern.match(sent):
        return False
    return True


def get_regex_sentences(num_sents, alphabet_map):
    alphabet = list(alphabet_map.keys())
    max_len = config.Data.max_len.int
    min_len = config.Data.min_len.int
    all_transformations = config.Grammar.all_transformations_enabled.boolean

    regex = '^' + config.Grammar.regex.str + '$'
    num_stars = regex.count('*')
    if num_stars == 0:
        raise Exception('must provide regex that contains at least one *')

    ranges = '{' + str(min_len // num_stars) + ',' + str(max_len // num_stars) + '}'
    regex = ranges.join(regex.split('*'))
    grammaticals = [list(rstr.xeger(regex)) for _ in range(num_sents // 2)]

    ungrammaticals = []
    left = num_sents // 2
    while left > 0:
        if all_transformations:
            random_lengths = np.random.randint(low=min_len, high=max_len, size=left)
            curr_ungrammaticals = [[random.choice(alphabet) for _ in range(length)] for length in random_lengths]
        else:
            sample = random.sample(grammaticals, left)
            curr_ungrammaticals = [random_trans(sentence, alphabet) for sentence in sample]
        ungrammaticals += list(filter(lambda sent: filter_by_regex(''.join(sent), regex),
                                      curr_ungrammaticals))
        left = (num_sents // 2) - len(ungrammaticals)

    data = np.array([np.array([alphabet_map[word] for word in sent]) for sent in grammaticals + ungrammaticals])
    labels = np.array([1] * len(grammaticals) + [0] * len(ungrammaticals))
    return data, labels


def random_trans(sentence, alphabet):
    result = copy(sentence)
    num_trans = np.random.randint(1, min(2, len(sentence)))
    for i in range(num_trans):
        trans = np.random.randint(2)
        ind = np.random.randint(len(sentence))
        if trans == 0:  # deletion
            result.pop(ind)
        else:
            new_char = np.random.choice(alphabet)
            if trans == 1:  # addition
                result.insert(ind, new_char)
            else:  # replacement
                result[ind] = new_char
    return result


def filter_out_grammatical_sentences(ungrammatical_sent, grammatical_sents):
    if tuple(ungrammatical_sent) in grammatical_sents:
        return False
    return True


def get_pos_num(pos):
    return pos_category_to_num[pos_category_map[pos]]


def read_conll_pos_file(path):
    """
    Takes a path to a file and returns a list of tags
    """
    sents = set()
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.add(tuple(curr))
                curr = []
            else:
                tokens = line.strip().split("\t")
                pos = tokens[3]
                curr.append(get_pos_num(pos))
    return sents


def get_data_alphabet():
    type_regex = config.Grammar.type_regex.boolean
    if type_regex:
        alphabet = config.Grammar.alphabet.lst
        alphabet_map = {a: i for i, a in enumerate(alphabet)}
    else:
        alphabet_map = pos_category_to_num
    return alphabet_map, {v: k for k, v in alphabet_map.items()}
