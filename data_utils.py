import random
import rstr
from random import shuffle
import numpy as np
from config import Config

config = Config()


def get_raw_data(DATA_AMOUNT):
    """Generate regex data - 50% grammatical and 50% ungrammatical"""
    max_sentence_length = config.Grammar.max_sentence_length.int
    min_sentence_length = config.Grammar.min_sentence_length.int
    semi_regex = config.Grammar.regular_expression.str
    regex = semi_regex + '{' + str(min_sentence_length) + ','+ str(max_sentence_length) + '}'
    alphabet = config.Grammar.alphabet.str

    grammatical_sents_list = [(rstr.xeger(regex), 1) for _ in range(int(np.floor(DATA_AMOUNT/2)))]
    transformation_type = config.Grammar.transformation_type.str

    if transformation_type == 'ALL_SENTENCES':
        ungrammatical_sents_list = []
        for _ in range(int(np.floor(DATA_AMOUNT / 2))):
            rand_len = random.randint(1, max_sentence_length)
            ungrammatical_sent = rstr.rstr(alphabet, rand_len)
            while ungrammatical_sent == rstr.xeger(semi_regex + '{' + str(int(len(ungrammatical_sent)/len(list(alphabet)))) + '}'):
                ungrammatical_sent = rstr.rstr(alphabet, rand_len)
            ungrammatical_sents_list.append((ungrammatical_sent, 0))

    else:
        ungrammatical_sents = [rstr.xeger(regex) for _ in range(int(np.floor(DATA_AMOUNT/2)))]
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
    raw_x, raw_y = get_raw_data(DATA_AMOUNT)
    percenta = int(DATA_AMOUNT * 0.7)
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