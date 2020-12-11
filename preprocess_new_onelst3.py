import tensorflow as tf
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import nltk
import string
import pronouncing
import random

# Gutenberg Dataset from: https://www.kaggle.com/terminate9298/gutenberg-poetry-dataset
GUTENBERG_DATASET_FILE = 'data/Gutenberg-Poetry.csv'  # Poetry from Gutenberg Project containing 2703086 Rows of Sentences

# Poetry Foundation Dataset from: https://www.kaggle.com/johnhallman/complete-poetryfoundationorg-dataset
POETRY_FOUNDATION_DATASET_FILE = 'data/kaggle_poem_dataset.csv'  # Dataset containing most poems found on poetryfoundation.org

# QUESTION: Should we stick to a single dataset since there are some inconsistencies?

def split_gutenberg(shuffle=False, test_size=0.2):
    """
    Poetry from Gutenberg projects contains 2703086 Rows of Sentences
    :param shuffle:
    :param test_size:
    :return: 2162468 Training sentences & 540618 Testing sentences
    """
    gutenberg_data = pandas.read_csv(GUTENBERG_DATASET_FILE)
    gutenberg_data = gutenberg_data[['s']].drop_duplicates(subset=['s'])
    if shuffle:
        gutenberg_data = gutenberg_data.sample(frac=1).reset_index(drop=True)
    gutenberg_train, gutenberg_test = train_test_split(gutenberg_data, test_size=test_size)
    return gutenberg_train, gutenberg_test


def split_poetry_foundation(shuffle=False, test_size=0.2):
    """
    Entire poems (15638) from poetryfoundation.org
    :param shuffle:
    :param test_size:
    :return: 12510 whole Training poems, 3128 whole Testing poems
    """
    poetry_foundation_data = pandas.read_csv(POETRY_FOUNDATION_DATASET_FILE)
    poetry_foundation_data = poetry_foundation_data.loc[poetry_foundation_data["Author"].isin(["William Shakespeare", "Emily Dickinson", "Robert Frost", "Alfred, Lord Tennyson", "William Butler Yeats", "William Wordsworth", "Walt Whitman"])]
    poetry_foundation_data = poetry_foundation_data[['Content']].drop_duplicates(subset='Content')
    poetry_foundation_data = np.asarray(poetry_foundation_data)[1:]
    if shuffle:
        poetry_foundation_data = poetry_foundation_data.sample(frac=1).reset_index(drop=True)
    words_lst = []
    for idx,line in enumerate(poetry_foundation_data):
        one_line = line[0].replace('\r', '').replace(u'\xa0', u'').lower()
        one_line = "".join([char for char in one_line if char not in string.punctuation])
        line_lst = one_line.split('\n')
        ss_line = [line1+' *' for line1 in line_lst]
        line_str = ''
        for e in ss_line:
            if e != ' *':
                line_str += e + ' '
        split_line = line_str.split(' ')
        split_line.remove('')
        while '' in split_line:
            split_line.remove('')
        words_lst += split_line
        if idx == 0:
            print(line[0])
            print(line_lst)
            print(ss_line)
            print(line_str)
            print(one_line)
            print(split_line)

    print(words_lst[:13])

    unique_words = list(set(words_lst))
    words_dict = dict((key, i) for i,key in enumerate(unique_words))

    # poetry_foundation_train, poetry_foundation_test = train_test_split(words_lst, test_size=test_size)

    poetry_foundation_train = words_lst[0:int(len(words_lst)*(1-test_size))]
    poetry_foundation_test = words_lst[int(len(words_lst)*(1-test_size)):]
    
    training_tokens = [words_dict.get(k) for k in poetry_foundation_train]
    testing_tokens = [words_dict.get(k) for k in poetry_foundation_test]

    print(training_tokens[:13])

    phenome_to_word_dict = {}
    phenome_lst = []
    # unique_words.remove('')
    for w in unique_words:
        # print('a'+w+'b')
        # print(pronouncing.phones_for_word(w))
        # if w == '':
        #     print('f')
        phenomes = pronouncing.phones_for_word(w)
        if phenomes != []:
            phenomes = phenomes[0].split(' ')
            phenome_to_word_dict[w] = phenomes
            phenome_lst += phenomes
        else:
            phenome_to_word_dict[w] = ''
            phenome_lst += ['']
    
    unique_phenomes = list(set(phenome_lst))
    phenome_dict = dict((key, i) for i,key in enumerate(unique_phenomes))
    print(len(phenome_dict))
    print(len(words_dict))

    phonemes_train = phenome_lst[0:int(len(phenome_lst)*(1-test_size))]
    phonemes_test = phenome_lst[int(len(phenome_lst)*(1-test_size)):]
    
    phonemes_training_tokens = [phenome_dict.get(k) for k in phonemes_train]
    phonemes_testing_tokens = [phenome_dict.get(k) for k in phonemes_test]

    word_to_tokphenome_dict = {}
    for w in unique_words:
        phenomes = pronouncing.phones_for_word(w)
        if phenomes != []:
            phenomes = phenomes[0].split(' ')
            tok_phenome = []
            for p in phenomes:
                sound = phenome_dict.get(p)
                tok_phenome.append(sound)
            word_to_tokphenome_dict[w] = tok_phenome
        else:
            word_to_tokphenome_dict[w] = [phenome_dict.get('')]
    print(len(word_to_tokphenome_dict))

    rhyming_arr = []
    common_words_raw = open('data/google-10000-english-usa.txt', 'r')
    common_words = common_words_raw.readlines()
    common_words_lst = []
    for line in common_words:
        arr = line.strip().split()
        common_words_lst += arr
    common_words_raw.close()
    for w in unique_words:
        rhymes = pronouncing.rhymes(w)
        if rhymes != []:
            for r in rhymes:
                rhyming_arr.append([w, r, 1])
                nonrhyme = random.choice(common_words_lst)
                if nonrhyme not in rhymes:
                    rhyming_arr.append([w, nonrhyme, 0])

    rhyming_arr = np.asarray(rhyming_arr)
    print(np.shape(rhyming_arr))

    return training_tokens, testing_tokens, words_dict, phonemes_training_tokens, phonemes_testing_tokens, phenome_dict


def get_data():
    """
    Read and parse train & test file line by line
    Tokenize sentences to build train & test data individually
    Construct Vocab Dictionary - uniques tokens from data as keys to unique integer values
    Vectorize train & test data based on Vocab Dictionary
    :param train_file: Training file path
    :param test_file: Testing file path
    :return:
    (1) Training Words in vectorized ID form
    (2) Testing Words in vectorized ID form
    (3) Map of indices & words "Dictionary"
    """
    # train_ids, test_ids, vocabulary_dict = [], [], {}
    # gutenberg_train, gutenberg_test = split_gutenberg()
    poetry_foundation_train, poetry_foundation_test, vocabulary_dict, phonemes_train, phonemes_test, phenome_dict = split_poetry_foundation()
    # print(np.asarray(poetry_foundation_train)[1])
    # train_data = pandas.concat([gutenberg_train, poetry_foundation_train])
    # train_data = gutenberg_train.append(poetry_foundation_test)
    # test_data = pandas.concat([gutenberg_test, poetry_foundation_test])
    # print(train_data)
    # print(test_data)
    # return train_ids, test_ids, vocabulary_dict
    return poetry_foundation_train, poetry_foundation_test, vocabulary_dict, phonemes_train, phonemes_test, phenome_dict


get_data()
