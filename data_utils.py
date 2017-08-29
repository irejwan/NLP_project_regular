import random
import re

import rstr
from random import shuffle
import numpy as np
from config import Config
from PCFG import PCFG

config = Config()


def get_raw_data(DATA_AMOUNT):
    """Generate regex data - 50% grammatical and 50% ungrammatical"""
    max_sentence_length = config.Grammar.max_sentence_length.int
    min_sentence_length = config.Grammar.min_sentence_length.int
    semi_regex = config.Grammar.regular_expression.str
    regex = semi_regex + '{' + str(min_sentence_length) + ',' + str(max_sentence_length) + '}'
    alphabet = config.Grammar.alphabet.lst

    grammatical_sents_list = [(rstr.xeger(regex), 1) for _ in range(int(np.floor(DATA_AMOUNT / 2)))]
    transformation_type = config.Grammar.transformation_type.str

    if transformation_type == 'ALL_SENTENCES':
        ungrammatical_sents_list = []
        for _ in range(int(np.floor(DATA_AMOUNT / 2))):
            rand_len = random.randint(1, max_sentence_length)
            ungrammatical_sent = rstr.rstr(alphabet, rand_len)
            while ungrammatical_sent == rstr.xeger(semi_regex + '{' + str(
                    int(len(ungrammatical_sent) / len(list(alphabet)))) + '}'):
                ungrammatical_sent = rstr.rstr(alphabet, rand_len)
            ungrammatical_sents_list.append((ungrammatical_sent, 0))

    else:
        ungrammatical_sents = [rstr.xeger(regex) for _ in range(int(np.floor(DATA_AMOUNT / 2)))]
        ungrammatical_sents_list = []
        for sent in ungrammatical_sents:
            if sent != '':
                if len(sent) > max_sentence_length:
                    sent = sent[:max_sentence_length]
                index = random.randint(0, len(sent) - 1)
                new_sent = sent[:index] + sent[index + 1:]
                ungrammatical_sents_list.append((new_sent, 0))

    total_sentences = [a for a in grammatical_sents_list + ungrammatical_sents_list if len(a[0]) > 0]
    shuffle(total_sentences)
    x = [list(map(int, list(a[0]))) for a in total_sentences]
    y = [a[1] for a in total_sentences]
    return x, y


def generate_sentences(DATA_AMOUNT):
    """Generate data and return it splitted to train, test and labels"""
    raw_x, raw_y = get_penn_pos_data(DATA_AMOUNT)
    # raw_x, raw_y = get_raw_data(DATA_AMOUNT)
    # raw_x, raw_y = get_1_star_2_star(DATA_AMOUNT)
    percenta = int(DATA_AMOUNT * 0.8)
    zipped = list(zip(raw_x, raw_y))
    random.shuffle(zipped)
    raw_x, raw_y = zip(*zipped)
    for i, j in zip(raw_x[:5], raw_y[:5]):
        print(i, j)

    X_train = np.array(raw_x[:percenta])
    y_train = np.array(raw_y[:percenta])
    X_test = np.array(raw_x[percenta:])
    y_test = np.array(raw_y[percenta:])

    X_train, y_train = zip(*(sorted(zip(X_train, y_train), key=lambda x: len(x[0]))))
    X_test, y_test = zip(*(sorted(zip(X_test, y_test), key=lambda x: len(x[0]))))
    X_train = np.array([np.array(x) for x in X_train])
    X_test = np.array([np.array(x) for x in X_test])
    y_train = np.array([np.array(x) for x in y_train])
    y_test = np.array([np.array(x) for x in y_test])
    return X_train, y_train, X_test, y_test


def get_1_star_2_star(total_num_of_sents):
    min_seq_len = config.Grammar.min_sentence_length.int
    max_seq_len = config.Grammar.max_sentence_length.int
    ns = np.random.randint(low=min_seq_len, high=max_seq_len // 2, size=total_num_of_sents // 2)
    grammaticals = list(map(lambda n: '12' * n, ns))
    ungrammaticals = []
    left = total_num_of_sents // 2
    while left > 0:
        curr_ungrammaticals = generate_random_strings(max_seq_len, left, alphabet=config.Grammar.alphabet.lst)
        ungrammaticals += list(filter(filter_out_1_star_2_star, curr_ungrammaticals))
        left = (total_num_of_sents // 2) - len(ungrammaticals)

    data = list(map(lambda sent: [int(s) for s in list(sent)], grammaticals + ungrammaticals))
    labels = np.array([1] * len(grammaticals) + [0] * len(ungrammaticals))
    return data, labels


def generate_random_strings(max_length, num_of_sents, alphabet):
    random_lengths = np.random.randint(low=1, high=max_length, size=num_of_sents)
    return [rstr.rstr(alphabet, length) for length in random_lengths]


def filter_out_1_star_2_star(sent):
    n = len(sent) // 2
    if '12' * n == sent:
        return False
    return True


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


def get_penn_pos_data(total_num_of_sents):
    grammatical_sents = read_conll_pos_file("../Penn_Treebank/dev.gold.conll")
    grammaticals = list(grammatical_sents)[:total_num_of_sents//2]

    ungrammaticals = []
    left = total_num_of_sents // 2
    alphabet = list(pos_category_to_num.keys())
    while left > 0:
        curr_ungrammaticals = generate_random_strings(config.Grammar.max_sentence_length.int, left, alphabet)
        curr_ungrammaticals = list(map(lambda sent: [pos_category_to_num[s] for s in list(sent)], curr_ungrammaticals))
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


def filter_out_sentences_that_match_the_pattern(sent, regex):
    pattern = re.compile(regex)
    if pattern.match(sent):
        return False
    return True


def get_simple_pos_data(num_of_sents):
    # todo: read regex from config
    regex = '^(Det )?(Adj ){0,3}Noun ((Prep )(Det )?(Adj ){0,3}Noun )?V (((Det )?(Adj ){0,3}Noun )((Prep )(Det )?(Adj ){0,3}Noun )?)?$'
    grammaticals = [rstr.xeger(regex).split(' ')[:-1] for _ in range(num_of_sents // 2)]
    max_seq_len = max([len(sent) for sent in grammaticals])
    ungrammaticals = []
    alphabet = ['Det ', 'Adj ', 'Noun ', 'Prep ', 'V ']
    # todo: make it more generic - parse it from regex
    alphbet_map = {a: i for i, a in enumerate(alphabet)}
    left = num_of_sents // 2
    while left > 0:
        curr_ungrammaticals = generate_random_strings(max_seq_len, left, alphabet)
        filtered = list(filter(lambda sent: filter_out_sentences_that_match_the_pattern(sent, regex),
                                      curr_ungrammaticals))
        ungrammaticals += [sent.split(' ')[:-1] for sent in filtered]
        left = (num_of_sents // 2) - len(ungrammaticals)
    data = grammaticals + ungrammaticals
    # todo: map to numbers
    labels = np.array([1] * len(grammaticals) + [0] * len(ungrammaticals))
    return data, labels


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


if __name__ == '__main__':
    print(list(zip(*get_simple_pos_data(100))))
